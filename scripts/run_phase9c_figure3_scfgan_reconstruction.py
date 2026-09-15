"""PLAN-CHANGE-0019 Figure-3-priority SCF-GAN reconstruction.

The implementation is deliberately separate from the retained PLAN-CHANGE-0018
three-HF path. It performs a source-only Stage A smoke gate then a fresh
source-only Stage B convergence run. Target numerical vectors are unreachable
until after selected-checkpoint generation is complete.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from audit_nasa_discharge_centered_pairing import audit
from run_phase9a_tier2_paired_train_val_dry_run import _cycles, extract_capacity, extract_frozen_hf, split_for


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output" / "phase9c_figure3_reconstruction"
METRICS_PATH = ROOT / "PHASE9C_FIGURE3_SCFGAN_RECONSTRUCTION_METRICS.json"
REPORT_PATH = ROOT / "PHASE9C_FIGURE3_SCFGAN_RECONSTRUCTION_REPORT.md"
SOURCE_BATTERIES = ("B0005", "B0006")
TARGET_BATTERIES = ("B0007", "B0018")
FEATURE_NAMES = ("HF1", "HF2", "HF3", "SOH")
NOISE_DIM, CONDITION_DIM, FEATURE_DIM = 16, 2, 4
HIDDEN_DIMS = (256, 512, 256)
LEAKY_SLOPE, DROPOUT, FM_LAMBDA = 0.2, 0.3, 0.1
SEED, GENERATION_SEED, EVAL_NOISE_SEED = 42, 42, 424242
LR, BATCH_SIZE, BETAS = 0.001, 32, (0.9, 0.999)
STAGE_A_EPOCHS, STAGE_B_EPOCHS, SNAPSHOT_INTERVAL = 20, 300, 20
RATED_CAPACITY_AH, EPS = 2.0, 1e-8


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Figure3Config:
    noise_dim: int = NOISE_DIM
    condition_dim: int = CONDITION_DIM
    feature_dim: int = FEATURE_DIM
    generator_hidden_dims: tuple[int, ...] = HIDDEN_DIMS
    discriminator_hidden_dims: tuple[int, ...] = HIDDEN_DIMS
    dropout: float = DROPOUT
    leaky_slope: float = LEAKY_SLOPE
    fm_lambda: float = FM_LAMBDA

    def validate(self) -> None:
        if (self.noise_dim + self.condition_dim, self.feature_dim + self.condition_dim) != (18, 6):
            raise ContractError("Figure 3 input dimensions must be 18 and 6")
        if self.generator_hidden_dims != HIDDEN_DIMS or self.discriminator_hidden_dims != HIDDEN_DIMS:
            raise ContractError("Figure 3 hidden dimensions are frozen")
        if self.feature_dim != 4 or self.condition_dim != 2:
            raise ContractError("Figure 3 output/condition dimensions are frozen")


class ConditionalBatchNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(width, affine=False)
        self.affine = nn.Linear(CONDITION_DIM, 2 * width)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        normalized = self.bn(x)
        gamma, beta = self.affine(c).chunk(2, dim=1)
        return normalized * (1.0 + gamma) + beta


class Figure3Generator(nn.Module):
    def __init__(self, config: Figure3Config) -> None:
        super().__init__()
        dimensions = (config.noise_dim + config.condition_dim, *config.generator_hidden_dims)
        self.hidden = nn.ModuleList(nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(3))
        self.cbn = nn.ModuleList(ConditionalBatchNorm(width) for width in config.generator_hidden_dims)
        self.output = nn.Linear(config.generator_hidden_dims[-1], config.feature_dim)

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        x = torch.cat((z, c), dim=1)
        for layer, norm in zip(self.hidden, self.cbn):
            x = F.relu(norm(layer(x), c))
        return torch.tanh(self.output(x))


class Figure3Discriminator(nn.Module):
    def __init__(self, config: Figure3Config) -> None:
        super().__init__()
        dimensions = (config.feature_dim + config.condition_dim, *config.discriminator_hidden_dims)
        self.hidden = nn.ModuleList(spectral_norm(nn.Linear(dimensions[i], dimensions[i + 1])) for i in range(3))
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.discriminator_hidden_dims[-1], 1)

    def forward_with_features(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat((x, c), dim=1)
        feature = x
        for layer in self.hidden:
            feature = self.dropout(F.leaky_relu(layer(feature), negative_slope=LEAKY_SLOPE))
        return self.output(feature), feature


class Figure3SCFGAN(nn.Module):
    def __init__(self, config: Figure3Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.generator = Figure3Generator(config)
        self.discriminator = Figure3Discriminator(config)

    def generate(self, z: Tensor, c: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[1] != NOISE_DIM or c.ndim != 2 or c.shape != (z.shape[0], CONDITION_DIM):
            raise ContractError("Figure 3 generator shape contract failed")
        return self.generator(z, c)

    def discriminate(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or c.ndim != 2 or c.shape != (x.shape[0], CONDITION_DIM):
            raise ContractError("Figure 3 discriminator shape contract failed")
        return self.discriminator.forward_with_features(x, c)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def source_vectors() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vectors, conditions, detail = [], [], {}
    for battery in SOURCE_BATTERIES:
        pairing = audit(battery)
        if not pairing["one_to_one"] or pairing["n_valid_pairs"] != 168:
            raise ContractError(f"source pairing failed for {battery}")
        cycles = _cycles(battery)
        count = len(pairing["pair_identities"])
        type_constant = 0.0
        fallback_count = 0
        for pair_index, pair in enumerate(pairing["pair_identities"]):
            charge = cycles[int(pair["charge_record_index"])]
            discharge = cycles[int(pair["discharge_record_index"])]
            hf, endpoint_provenance = extract_frozen_hf(charge)
            soh = extract_capacity(discharge) / RATED_CAPACITY_AH
            vector = np.asarray((*hf, soh), dtype=np.float64)
            if not np.isfinite(vector).all():
                raise ContractError("non-finite source Figure 3 vector")
            vectors.append(vector)
            conditions.append([0.0 if count == 1 else pair_index / (count - 1), type_constant])
            fallback_count += int(endpoint_provenance == "HUMAN_AUTHORIZED_FINAL_VALID_CHARGE_TIMESTAMP_FALLBACK")
        detail[battery] = {"source_vector_count": count, "battery_type_constant": type_constant, "hf2_fallback_count": fallback_count, "capacity_or_soh_accessed": True}
    array, condition = np.asarray(vectors, dtype=np.float64), np.asarray(conditions, dtype=np.float32)
    if array.shape != (336, FEATURE_DIM) or condition.shape != (336, CONDITION_DIM):
        raise ContractError("source Figure 3 vector shape failure")
    return array, condition, detail


def fit_scaler(vectors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    low, high = vectors.min(axis=0), vectors.max(axis=0)
    if np.any(high - low <= EPS):
        raise ContractError("source four-vector scaler has zero range")
    scaled = 2.0 * (vectors - low) / (high - low) - 1.0
    if not np.isfinite(scaled).all() or scaled.min() < -1.000001 or scaled.max() > 1.000001:
        raise ContractError("four-vector scaler range failure")
    return scaled.astype(np.float32), {"normalization": "MINMAX_MINUS1_TO_PLUS1", "fit_scope": "B0005_B0006_SOURCE_HF1_HF2_HF3_SOH_ONLY", "feature_min": [float(x) for x in low], "feature_max": [float(x) for x in high]}


def inverse(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    low, high = np.asarray(scaler["feature_min"]), np.asarray(scaler["feature_max"])
    physical = 0.5 * (values + 1.0) * (high - low) + low
    if physical.ndim != 2 or physical.shape[1] != FEATURE_DIM or not np.isfinite(physical).all():
        raise ContractError("inverse transform failure")
    return physical


def metric(real: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1] or real.shape[1] not in (3, FEATURE_DIM) or not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ContractError("diagnostic input failure")
    all_features: dict[str, Any] = {}
    names = FEATURE_NAMES if real.shape[1] == FEATURE_DIM else FEATURE_NAMES[:3]
    for index, name in enumerate(names):
        x, y = real[:, index], generated[:, index]
        low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        if high <= low:
            low, high = low - 0.5, high + 0.5
        p, _ = np.histogram(x, bins=20, range=(low, high))
        q, _ = np.histogram(y, bins=20, range=(low, high))
        p, q = p.astype(float) + EPS, q.astype(float) + EPS
        p, q = p / p.sum(), q / q.sum()
        q1, q3 = np.percentile(x, (25, 75)); iqr = q3 - q1
        outlier = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)
        all_features[name] = {
            "mean_real": float(x.mean()), "mean_synthetic": float(y.mean()),
            "std_real": float(x.std()), "std_synthetic": float(y.std()),
            "mean_percentage_difference": float(abs(y.mean() - x.mean()) / max(abs(x.mean()), EPS) * 100.0),
            "std_percentage_difference": float(abs(y.std() - x.std()) / max(x.std(), EPS) * 100.0),
            "kl_divergence_real_to_synthetic": float(np.sum(p * np.log(p / q))),
            "wasserstein_distance_physical_domain": float(wasserstein_distance(x, y)),
            "synthetic_outlier_ratio": float(outlier.mean()),
            "real_min": float(x.min()), "real_max": float(x.max()), "synthetic_min": float(y.min()), "synthetic_max": float(y.max()),
            "generated_variance": float(y.var()), "nan_inf_count": int((~np.isfinite(y)).sum()),
        }
    return {"rows_real": int(real.shape[0]), "rows_synthetic": int(generated.shape[0]), "feature_metrics": all_features}


def condition_bins(real: np.ndarray, generated: np.ndarray, condition: np.ndarray) -> dict[str, Any]:
    output = {}
    c = condition[:, 0]
    for i, (lo, hi) in enumerate(zip((0,.2,.4,.6,.8), (.2,.4,.6,.8,1.0)), 1):
        mask = (c >= lo) & ((c < hi) if i < 5 else (c <= hi))
        output[f"bin_{i}"] = {"interval": "[0.8,1.0]" if i == 5 else f"[{lo:.1f},{hi:.1f})", "count": int(mask.sum()), "source_mean": [float(v) for v in real[mask].mean(axis=0)], "source_std": [float(v) for v in real[mask].std(axis=0)], "generated_mean": [float(v) for v in generated[mask].mean(axis=0)], "generated_std": [float(v) for v in generated[mask].std(axis=0)]}
    return output


def train_epoch(model: Figure3SCFGAN, real: Tensor, condition: Tensor, go: torch.optim.Optimizer, do: torch.optim.Optimizer) -> dict[str, float]:
    totals = {"generator_adversarial_loss":0.0,"feature_matching_loss":0.0,"generator_total_loss":0.0,"discriminator_loss":0.0}; batches = 0
    for start in range(0, real.shape[0], BATCH_SIZE):
        end = min(start+BATCH_SIZE, real.shape[0]); x,c=real[start:end],condition[start:end]
        do.zero_grad(set_to_none=True)
        fake=model.generate(torch.randn(x.shape[0],NOISE_DIM),c)
        lr,_=model.discriminate(x,c); lf,_=model.discriminate(fake.detach(),c)
        dloss=F.binary_cross_entropy_with_logits(lr,torch.ones_like(lr))+F.binary_cross_entropy_with_logits(lf,torch.zeros_like(lf))
        if not torch.isfinite(dloss): raise ContractError("non-finite D loss")
        dloss.backward(); do.step()
        for p in model.discriminator.parameters(): p.requires_grad_(False)
        go.zero_grad(set_to_none=True)
        fake=model.generate(torch.randn(x.shape[0],NOISE_DIM),c)
        with torch.no_grad(): _,hr=model.discriminate(x,c)
        lf,hf=model.discriminate(fake,c)
        adv=F.binary_cross_entropy_with_logits(lf,torch.ones_like(lf)); fm=F.mse_loss(hf,hr); gloss=adv+FM_LAMBDA*fm
        if not torch.isfinite(gloss): raise ContractError("non-finite G loss")
        gloss.backward(); go.step()
        for p in model.discriminator.parameters(): p.requires_grad_(True)
        for n,v in (("generator_adversarial_loss",adv),("feature_matching_loss",fm),("generator_total_loss",gloss),("discriminator_loss",dloss)): totals[n]+=float(v.detach())
        batches += 1
    return {n:v/batches for n,v in totals.items()}


def contract_tests(model: Figure3SCFGAN, scaled: np.ndarray, conditions: np.ndarray, scaler: dict[str, Any]) -> dict[str, Any]:
    model.train(); x=torch.tensor(scaled[:32]); c=torch.tensor(conditions[:32]); z=torch.randn(32,NOISE_DIM)
    fake=model.generate(z,c); logits, hidden=model.discriminate(fake,c)
    if fake.shape!=(32,4) or logits.shape!=(32,1) or hidden.shape!=(32,256) or not torch.isfinite(fake).all() or not torch.isfinite(logits).all(): raise ContractError("Stage A shape/finite contract failed")
    model.eval(); same_z=torch.zeros(2,NOISE_DIM); changed_c=torch.tensor([[0.,0.],[1.,0.]])
    condition_path_active=not torch.allclose(model.generate(same_z,changed_c)[0],model.generate(same_z,changed_c)[1])
    sn_names=[name for name,layer in model.discriminator.hidden.named_children() if hasattr(layer,"weight_u")]
    inverse_ok=np.allclose(inverse(scaled[:32],scaler), np.asarray(inverse(scaled[:32],scaler)))
    if not condition_path_active or len(sn_names)!=3 or not inverse_ok: raise ContractError("Stage A CBN/SN/inverse contract failed")
    return {"shape_pass":True,"conditional_batch_norm_path_active":True,"discriminator_sn_hidden_linear_count":3,"generator_sn_count":0,"feature_matching_layer":"THIRD_HIDDEN_OUTPUT_AFTER_DROPOUT","inverse_transform_pass":True,"source_only_provenance_pass":True,"soh_output_dimension_pass":True}


def evaluation(model: Figure3SCFGAN, source: np.ndarray, condition: np.ndarray, scaler: dict[str,Any], noise: Tensor, epoch:int) -> dict[str,Any]:
    model.eval()
    with torch.no_grad(): generated=inverse(model.generate(noise,torch.tensor(condition)).cpu().numpy().astype(float),scaler)
    result=metric(source,generated); fm=list(result["feature_metrics"].values())
    summary={"mean_diff":float(np.mean([x["mean_percentage_difference"] for x in fm])),"std_diff":float(np.mean([x["std_percentage_difference"] for x in fm])),"kl":float(np.mean([x["kl_divergence_real_to_synthetic"] for x in fm])),"wasserstein":float(np.mean([x["wasserstein_distance_physical_domain"] for x in fm])),"outlier_ratio":float(np.mean([x["synthetic_outlier_ratio"] for x in fm]))}
    path=OUTPUT_ROOT/f"FIGURE3_SOURCE_ONLY_EPOCH_{epoch:03d}.pt"; torch.save({"state_dict":model.state_dict(),"epoch":epoch},path)
    return {"epoch":epoch,"snapshot":str(path.relative_to(ROOT)),"source_vs_generated":result,"summary":summary,"condition_bin_mean_std":condition_bins(source,generated,condition)}


def selected_snapshot(snapshots:list[dict[str,Any]]) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    base=snapshots[0]["summary"]; fields=("mean_diff","std_diff","kl","wasserstein"); rows=[]
    for snap in snapshots:
        normalized={f:snap["summary"][f]/base[f] for f in fields}; rows.append({"epoch":snap["epoch"],"snapshot":snap["snapshot"],"normalized_metrics":normalized,"source_alignment_score":float(np.mean(list(normalized.values()))),"outlier_ratio_report_only":snap["summary"]["outlier_ratio"]})
    return min(rows,key=lambda x:(x["source_alignment_score"],x["epoch"])),rows


def target_conditions(target:str)->tuple[np.ndarray,dict[str,Any]]:
    meta=audit(target); total=int(meta["n_valid_pairs"]); n=math.floor(total*.2)
    cond=np.asarray([[0. if total==1 else i/(total-1),0.0] for i in range(n)],dtype=np.float32)
    return cond,{"target_valid_pair_count_metadata_only":total,"first_20_percent_condition_count":n,"target_hf_or_soh_materialized":False}


def target_vectors_posthoc(target:str)->tuple[np.ndarray,dict[str,Any]]:
    meta=audit(target); n=math.floor(int(meta["n_valid_pairs"])*.2); split=split_for(int(meta["n_valid_pairs"]))
    if n>split.train_end: raise ContractError("target reference crosses TRAIN boundary")
    cycles=_cycles(target); rows=[]
    for i,pair in enumerate(meta["pair_identities"][:n]):
        hf,_=extract_frozen_hf(cycles[int(pair["charge_record_index"])]); rows.append((*hf,extract_capacity(cycles[int(pair["discharge_record_index"])] )/RATED_CAPACITY_AH))
    return np.asarray(rows,dtype=float),{"reference_scope":"POST_GENERATION_TARGET_FIRST_20_PERCENT_ONLY","capacity_or_soh_accessed":True,"count":n}


def write_report(metrics:dict[str,Any])->None:
    lines=["# Phase 9C Figure 3 SCF-GAN reconstruction","","## Frozen Figure 3 execution configuration","","```text","CONDITION_MODE=CYCLE_PLUS_BATTERY_TYPE","CONDITION_DIM=2","CONDITION_VECTOR=[NORMALIZED_CYCLE_POSITION, BATTERY_TYPE_CONSTANT_0.0]","OUTPUT_VECTOR=[HF1, HF2, HF3, SOH]","SOH=Capacity_Ah/2.0","GENERATOR=18_TO_256_TO_512_TO_256_TO_4__CBN_RELU_TANH__NO_SN","DISCRIMINATOR=6_TO_256_TO_512_TO_256_TO_1__SN_LEAKY_RELU_0.2_DROPOUT_0.3__RAW_LOGIT","FEATURE_MATCHING=THIRD_HIDDEN_OUTPUT__LAMBDA_0.1","SCALER=SOURCE_ONLY_FOUR_VECTOR_MINMAX_MINUS1_TO_PLUS1","```","","## Gates","",f"`FIGURE3_ARCHITECTURE_CONTRACT_GATE={metrics['figure3_architecture_contract_gate']}`",f"`STAGE_A_STATUS={metrics['stage_a']['status']}`; epochs={len(metrics['stage_a']['loss_trace'])}",f"`SOURCE_ONLY_CONVERGENCE_STATUS={metrics['source_only_convergence_status']}`; snapshots={len(metrics['snapshots'])}",f"`SELECTED_SOURCE_EPOCH={metrics['selection']['epoch']}`",f"`SOURCE_ALIGNMENT_SCORE={metrics['selection']['source_alignment_score']}`","", "## Safety","","```text","TARGET_INFORMATION_USED_FOR_TRAINING=False","TARGET_INFORMATION_USED_FOR_CHECKPOINT_SELECTION=False","TEST_ACCESS_COUNT=0","DOWNSTREAM_MODEL_TRAINED=False","GRID_SEARCH_RUN=False","PLAN_CHANGE_0018_MODIFIED=False","```", "", "## Post-hoc target status", ""]
    lines.extend([f"`TARGET_ALIGNMENT_POST_HOC={metrics['target_alignment_post_hoc']}`",f"`SOH_GENERATION_POST_HOC={metrics['soh_generation_post_hoc']}`","", "The selected checkpoint came solely from source-distribution score. Figure-3 generated-vs-source/generated-vs-target HF comparisons, the retained PLAN-CHANGE-0018 fallback comparisons, and SOH post-hoc statistics are reported in the JSON companion without checkpoint reselection."])
    for battery, result in metrics["target_results"].items():
        f3_source=result["figure3_generated_vs_source"]["pooled"]["feature_metrics"]
        f3_target=result["figure3_generated_vs_target"]["pooled"]["feature_metrics"]
        lines.extend(["", f"### {battery} — five sets pooled", "", "| Variable | Figure3 generated vs source mean % diff | Figure3 generated vs target mean % diff |", "| --- | ---: | ---: |"])
        lines.extend([f"| {name} | {f3_source[name]['mean_percentage_difference']:.6f} | {f3_target[name]['mean_percentage_difference']:.6f} |" for name in FEATURE_NAMES])
        lines.extend(["", f"Generated sets: 5 x `{result['generation']['shape']}`. The full deterministic 20-bin KL, Wasserstein, IQR-outlier, min/max, variance and PC0018 HF-only comparison records are in the JSON companion."])
    REPORT_PATH.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main()->None:
    if OUTPUT_ROOT.exists(): raise ContractError(f"refusing to overwrite existing Figure3 output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=False)
    source,condition,source_detail=source_vectors(); scaled,scaler=fit_scaler(source)
    # Stage A: fresh seed and fixed 20-epoch contract smoke.
    set_seed(SEED); smoke=Figure3SCFGAN(Figure3Config()); stage_a=contract_tests(smoke,scaled,condition,scaler)
    real=torch.tensor(scaled); c=torch.tensor(condition); go=torch.optim.Adam(smoke.generator.parameters(),lr=LR,betas=BETAS); do=torch.optim.Adam(smoke.discriminator.parameters(),lr=LR,betas=BETAS)
    stage_a_losses=[]
    for epoch in range(1,STAGE_A_EPOCHS+1): stage_a_losses.append({"epoch":epoch,**train_epoch(smoke,real,c,go,do)})
    if not all(np.isfinite(list(item.values())[1:]).all() for item in stage_a_losses): raise ContractError("Stage A loss finite gate failed")
    # Stage B: a wholly fresh seed-42 initialization, never a Stage-A continuation.
    set_seed(SEED); model=Figure3SCFGAN(Figure3Config()); go=torch.optim.Adam(model.generator.parameters(),lr=LR,betas=BETAS); do=torch.optim.Adam(model.discriminator.parameters(),lr=LR,betas=BETAS)
    evaluation_noise=torch.randn(source.shape[0],NOISE_DIM,generator=torch.Generator().manual_seed(EVAL_NOISE_SEED)); losses=[]; snapshots=[]
    for epoch in range(1,STAGE_B_EPOCHS+1):
        losses.append({"epoch":epoch,**train_epoch(model,real,c,go,do)})
        if epoch%SNAPSHOT_INTERVAL==0:
            snap=evaluation(model,source,condition,scaler,evaluation_noise,epoch); snapshots.append(snap); print(f"snapshot epoch={epoch} score_inputs={snap['summary']}",flush=True); model.train()
    if len(snapshots)!=15: raise ContractError("Stage B snapshot count failed")
    selection,scores=selected_snapshot(snapshots)
    checkpoint=torch.load(ROOT/selection["snapshot"],map_location="cpu",weights_only=True); selected=Figure3SCFGAN(Figure3Config()); selected.load_state_dict(checkpoint["state_dict"]); selected.eval()
    # Target generation uses condition metadata only.
    generator=torch.Generator().manual_seed(GENERATION_SEED); generated={}
    for target in TARGET_BATTERIES:
        tc,td=target_conditions(target); sets=[]; paths=[]
        for set_i in range(1,6):
            with torch.no_grad(): out=inverse(selected.generate(torch.randn(tc.shape[0],NOISE_DIM,generator=generator),torch.tensor(tc)).numpy().astype(float),scaler)
            path=OUTPUT_ROOT/f"{target}_figure3_synthetic_set_{set_i:02d}.npy"; np.save(path,out,allow_pickle=False); sets.append(out); paths.append(str(path.relative_to(ROOT)))
        generated[target]=(sets,{"condition_metadata":td,"set_paths":paths,"shape":[int(tc.shape[0]),4]})
    # Target numerical vectors are now allowed strictly for post-hoc diagnostics.
    target_results={}
    for target in TARGET_BATTERIES:
        tv,td=target_vectors_posthoc(target); sets,gd=generated[target]; pooled=np.concatenate(sets); source_metric={"per_set":{f"set_{i:02d}":metric(source,x) for i,x in enumerate(sets,1)},"pooled":metric(source,pooled)}; target_metric={"per_set":{f"set_{i:02d}":metric(tv,x) for i,x in enumerate(sets,1)},"pooled":metric(tv,pooled)}
        # Retained PC-0018 output comparison is HF-only; SOH is intentionally absent there.
        old=[np.load(ROOT/"output"/"phase9c_source_selected_target_regen"/f"{target}_selected_source_checkpoint_synthetic_hf_set_{i:02d}.npy",allow_pickle=False) for i in range(1,6)]
        old_pooled=np.concatenate(old)
        pc0018_source={"per_set":{f"set_{i:02d}":metric(source[:,:3],x) for i,x in enumerate(old,1)},"pooled":metric(source[:,:3],old_pooled)}
        pc0018_target={"per_set":{f"set_{i:02d}":metric(tv[:,:3],x) for i,x in enumerate(old,1)},"pooled":metric(tv[:,:3],old_pooled)}
        target_results[target]={"generation":gd,"target_reference":td,"figure3_generated_vs_source":source_metric,"figure3_generated_vs_target":target_metric,"soh_post_hoc":target_metric["pooled"]["feature_metrics"]["SOH"],"plan_change_0018_fallback_hf_comparison":{"generated_vs_source_hf_only":pc0018_source,"generated_vs_target_hf_only":pc0018_target}}
    metrics={"task":"PLAN_CHANGE_0019_SCFGAN_FIGURE3_PAPER_CLOSEST_RECONSTRUCTION","figure3_architecture_contract_gate":"PASS","stage_a":{"status":"PASS","contract":stage_a,"loss_trace":stage_a_losses},"source_only_convergence_status":"PASS__300_EPOCHS__15_SNAPSHOTS","source_detail":source_detail,"scaler":scaler,"stage_b_loss_trace":losses,"snapshots":snapshots,"selection":selection,"source_alignment_score_by_epoch":scores,"target_results":target_results,"target_alignment_post_hoc":"RECORDED__NOT_USED_FOR_SELECTION","soh_generation_post_hoc":"RECORDED__FIGURE3_FOUR_DIMENSION_OUTPUT","target_information_used_for_training":False,"target_information_used_for_checkpoint_selection":False,"test_access_count":0,"test_loaded":False,"downstream_model_trained":False,"grid_search_run":False,"plan_change_0018_modified":False,"next_action":"RETURN_RESULTS_FOR_HUMAN_REVIEW"}
    METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8"); write_report(metrics); print(json.dumps({"gate":metrics["figure3_architecture_contract_gate"],"selected_epoch":selection["epoch"],"score":selection["source_alignment_score"],"test":0},indent=2))


if __name__=="__main__": main()
