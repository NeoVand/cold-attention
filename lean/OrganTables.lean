/-
OrganTables.lean (generated from the trained networks' tables; do not edit).

The trained kinship organs of layout `within` (two direction classes) as finite tables, each with
  * `sound_<run>` : the soundness check, by `decide`;
  * `cert_<run>`  : the completeness certificate check, by `decide`;
  * `correct_<run>` : the every-size theorem `Organ.within_correct` for this trained network.
Table entry: tbl[pattern] has bit 3 * (4 L0 + 2 L1 + L2) + channel set iff that test passes.
-/
import OrganCert

namespace Organ

/-- kinship/qwen2.5-1.5b/within/seed0: smallest |score| over all cases 0.0013; on the patterns that occur, with no kept bit, 0.1297. -/
def net_kinship_1p5b_within_seed0 : Net := { tbl := [16748032, 16777065, 15895040, 16226888, 12648448, 16747008, 8388608, 10519040, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 8388608, 16157184, 8388608, 10650112, 0, 8388608, 0, 0, 16777215, 16777215, 16777213, 16777213, 16777214, 16777215, 16772920, 16777085, 8388608, 16157184, 8388608, 10519040, 0, 8388608, 0, 0, 16777215, 16777215, 16777215, 16777213, 16773054, 16777213, 16772912, 16777084, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 14778880, 16751176, 10485760, 16155144, 4194304, 14713344, 0, 2097152, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16747008, 16751176, 10519040, 11967048, 12582912, 16220672, 0, 10519040, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16772904], rd := 236 }

def cert_data_kinship_1p5b_within_seed0 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, true⟩, Rl := ⟨false, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_1p5b_within_seed0 : checkSound net_kinship_1p5b_within_seed0 = true := by decide

theorem cert_kinship_1p5b_within_seed0 : checkCert net_kinship_1p5b_within_seed0 within cert_data_kinship_1p5b_within_seed0 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_1p5b_within_seed0 : net_kinship_1p5b_within_seed0.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-1.5b/within/seed0 is right on every well-formed instance of every size. -/
theorem correct_kinship_1p5b_within_seed0 {α : Type} (I : Inst α) (R : Run net_kinship_1p5b_within_seed0 I.pat) :
    (∀ t, net_kinship_1p5b_within_seed0.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_1p5b_within_seed0.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_1p5b_within_seed0 cert_kinship_1p5b_within_seed0 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_1p5b_within_seed0 {α : Type} (I : Inst α) (R : Run net_kinship_1p5b_within_seed0 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_1p5b_within_seed0.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_1p5b_within_seed0 R hno t

#print axioms correct_kinship_1p5b_within_seed0
#print axioms no_false_yes_kinship_1p5b_within_seed0

/-- kinship/qwen2.5-1.5b/within/seed1: smallest |score| over all cases 0.0016; on the patterns that occur, with no kept bit, 0.3002. -/
def net_kinship_1p5b_within_seed1 : Net := { tbl := [16748032, 16177960, 6324224, 2392576, 8388608, 8519680, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16773052, 16768876, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 0, 0, 0, 0, 0, 0, 0, 16777215, 16777213, 16777148, 16768876, 16510388, 16764716, 14353408, 15894528, 0, 2097152, 0, 0, 0, 0, 0, 0, 16777215, 16777213, 16777148, 16768876, 16510388, 16764732, 14371840, 15911200, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16764456, 16178024, 6324224, 10781184, 8388608, 10616832, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16222720, 11983368, 2129920, 2392584, 0, 8388608, 0, 0, 16777215, 16777215, 16777215, 16777213, 16777215, 16777213, 16777150, 16768877, 16745984], rd := 248 }

def cert_data_kinship_1p5b_within_seed1 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, false⟩, Rl := ⟨false, false, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_1p5b_within_seed1 : checkSound net_kinship_1p5b_within_seed1 = true := by decide

theorem cert_kinship_1p5b_within_seed1 : checkCert net_kinship_1p5b_within_seed1 within cert_data_kinship_1p5b_within_seed1 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_1p5b_within_seed1 : net_kinship_1p5b_within_seed1.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-1.5b/within/seed1 is right on every well-formed instance of every size. -/
theorem correct_kinship_1p5b_within_seed1 {α : Type} (I : Inst α) (R : Run net_kinship_1p5b_within_seed1 I.pat) :
    (∀ t, net_kinship_1p5b_within_seed1.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_1p5b_within_seed1.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_1p5b_within_seed1 cert_kinship_1p5b_within_seed1 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_1p5b_within_seed1 {α : Type} (I : Inst α) (R : Run net_kinship_1p5b_within_seed1 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_1p5b_within_seed1.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_1p5b_within_seed1 R hno t

#print axioms correct_kinship_1p5b_within_seed1
#print axioms no_false_yes_kinship_1p5b_within_seed1

/-- kinship/qwen2.5-1.5b/within/seed2: smallest |score| over all cases 0.0047; on the patterns that occur, with no kept bit, 0.0864. -/
def net_kinship_1p5b_within_seed2 : Net := { tbl := [16745984, 16777215, 16678912, 16777212, 15728640, 16777208, 8388608, 16764704, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 15728640, 16777208, 15728640, 16777128, 8388608, 16764704, 8388608, 16402720, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777213, 16777215, 15859712, 16777212, 13631488, 16777148, 8388608, 16764704, 8388608, 16468256, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16154624, 16777208, 15859712, 16777208, 8388608, 16764704, 8388608, 16764704, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777213, 16777215, 16748032, 16777215, 16449536, 16777212, 13631488, 16777212, 8388608, 16764708, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777176], rd := 238 }

def cert_data_kinship_1p5b_within_seed2 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, true⟩, Rl := ⟨true, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_1p5b_within_seed2 : checkSound net_kinship_1p5b_within_seed2 = true := by decide

theorem cert_kinship_1p5b_within_seed2 : checkCert net_kinship_1p5b_within_seed2 within cert_data_kinship_1p5b_within_seed2 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_1p5b_within_seed2 : net_kinship_1p5b_within_seed2.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-1.5b/within/seed2 is right on every well-formed instance of every size. -/
theorem correct_kinship_1p5b_within_seed2 {α : Type} (I : Inst α) (R : Run net_kinship_1p5b_within_seed2 I.pat) :
    (∀ t, net_kinship_1p5b_within_seed2.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_1p5b_within_seed2.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_1p5b_within_seed2 cert_kinship_1p5b_within_seed2 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_1p5b_within_seed2 {α : Type} (I : Inst α) (R : Run net_kinship_1p5b_within_seed2 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_1p5b_within_seed2.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_1p5b_within_seed2 R hno t

#print axioms correct_kinship_1p5b_within_seed2
#print axioms no_false_yes_kinship_1p5b_within_seed2

/-- kinship/qwen2.5-7b/per_sentence_within/seed0: smallest |score| over all cases 0.0036; on the patterns that occur, with no kept bit, 0.1348. -/
def net_kinship_7b_per_sentence_within_seed0 : Net := { tbl := [16748032, 16777065, 15895040, 16226888, 12648448, 16747008, 8388608, 10519040, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 8388608, 16157184, 8388608, 10650112, 0, 8388608, 0, 0, 16777215, 16777215, 16777213, 16777213, 16777214, 16777215, 16772920, 16777085, 8388608, 16157184, 8388608, 10519040, 0, 8388608, 0, 0, 16777215, 16777215, 16777215, 16777213, 16773054, 16777213, 16772912, 16777084, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 14778880, 16751176, 10485760, 16155144, 4194304, 14713344, 0, 2097152, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16747008, 16751176, 10519040, 11967048, 12582912, 16220672, 0, 10519040, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16772904], rd := 236 }

def cert_data_kinship_7b_per_sentence_within_seed0 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, true⟩, Rl := ⟨false, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_per_sentence_within_seed0 : checkSound net_kinship_7b_per_sentence_within_seed0 = true := by decide

theorem cert_kinship_7b_per_sentence_within_seed0 : checkCert net_kinship_7b_per_sentence_within_seed0 within cert_data_kinship_7b_per_sentence_within_seed0 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_7b_per_sentence_within_seed0 : net_kinship_7b_per_sentence_within_seed0.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-7b/per_sentence_within/seed0 is right on every well-formed instance of every size. -/
theorem correct_kinship_7b_per_sentence_within_seed0 {α : Type} (I : Inst α) (R : Run net_kinship_7b_per_sentence_within_seed0 I.pat) :
    (∀ t, net_kinship_7b_per_sentence_within_seed0.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_7b_per_sentence_within_seed0.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_7b_per_sentence_within_seed0 cert_kinship_7b_per_sentence_within_seed0 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_7b_per_sentence_within_seed0 {α : Type} (I : Inst α) (R : Run net_kinship_7b_per_sentence_within_seed0 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_7b_per_sentence_within_seed0.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_7b_per_sentence_within_seed0 R hno t

#print axioms correct_kinship_7b_per_sentence_within_seed0
#print axioms no_false_yes_kinship_7b_per_sentence_within_seed0

/-- kinship/qwen2.5-7b/per_sentence_within/seed1: smallest |score| over all cases 0.0002; on the patterns that occur, with no kept bit, 0.3380. -/
def net_kinship_7b_per_sentence_within_seed1 : Net := { tbl := [16748032, 16177960, 6324224, 10781184, 8388608, 8519680, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16777148, 16768876, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 2097152, 0, 0, 0, 0, 0, 0, 16777215, 16777215, 16777148, 16768876, 16510900, 16764716, 14355456, 15910912, 0, 2097152, 0, 0, 0, 0, 0, 0, 16777215, 16777215, 16777148, 16777084, 16510388, 16772924, 14371840, 15911200, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16764456, 16243560, 6324224, 10781184, 8388608, 10616832, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16222720, 16177672, 2129920, 2392576, 0, 10485760, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777213, 16777150, 16777085, 16748040], rd := 248 }

def cert_data_kinship_7b_per_sentence_within_seed1 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, false⟩, Rl := ⟨false, false, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_per_sentence_within_seed1 : checkSound net_kinship_7b_per_sentence_within_seed1 = true := by decide

theorem cert_kinship_7b_per_sentence_within_seed1 : checkCert net_kinship_7b_per_sentence_within_seed1 within cert_data_kinship_7b_per_sentence_within_seed1 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_7b_per_sentence_within_seed1 : net_kinship_7b_per_sentence_within_seed1.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-7b/per_sentence_within/seed1 is right on every well-formed instance of every size. -/
theorem correct_kinship_7b_per_sentence_within_seed1 {α : Type} (I : Inst α) (R : Run net_kinship_7b_per_sentence_within_seed1 I.pat) :
    (∀ t, net_kinship_7b_per_sentence_within_seed1.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_7b_per_sentence_within_seed1.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_7b_per_sentence_within_seed1 cert_kinship_7b_per_sentence_within_seed1 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_7b_per_sentence_within_seed1 {α : Type} (I : Inst α) (R : Run net_kinship_7b_per_sentence_within_seed1 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_7b_per_sentence_within_seed1.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_7b_per_sentence_within_seed1 R hno t

#print axioms correct_kinship_7b_per_sentence_within_seed1
#print axioms no_false_yes_kinship_7b_per_sentence_within_seed1

/-- kinship/qwen2.5-7b/per_sentence_within/seed2: smallest |score| over all cases 0.0022; on the patterns that occur, with no kept bit, 0.2141. -/
def net_kinship_7b_per_sentence_within_seed2 : Net := { tbl := [16748032, 16777215, 16678912, 16777212, 15728640, 16777208, 8388608, 16764704, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 15728640, 16777208, 14680064, 16777144, 8388608, 16764672, 8388608, 16402688, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777212, 16777215, 15859712, 16777212, 13631488, 16777212, 8388608, 16764704, 8388608, 16469280, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16678912, 16777208, 15728640, 16777208, 8388608, 16764704, 8388608, 16764704, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777213, 16777215, 16748032, 16777215, 16449536, 16777213, 13631488, 16777212, 8388608, 16764708, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777176], rd := 238 }

def cert_data_kinship_7b_per_sentence_within_seed2 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, true⟩, Rl := ⟨true, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_per_sentence_within_seed2 : checkSound net_kinship_7b_per_sentence_within_seed2 = true := by decide

theorem cert_kinship_7b_per_sentence_within_seed2 : checkCert net_kinship_7b_per_sentence_within_seed2 within cert_data_kinship_7b_per_sentence_within_seed2 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_7b_per_sentence_within_seed2 : net_kinship_7b_per_sentence_within_seed2.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-7b/per_sentence_within/seed2 is right on every well-formed instance of every size. -/
theorem correct_kinship_7b_per_sentence_within_seed2 {α : Type} (I : Inst α) (R : Run net_kinship_7b_per_sentence_within_seed2 I.pat) :
    (∀ t, net_kinship_7b_per_sentence_within_seed2.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_7b_per_sentence_within_seed2.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_7b_per_sentence_within_seed2 cert_kinship_7b_per_sentence_within_seed2 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_7b_per_sentence_within_seed2 {α : Type} (I : Inst α) (R : Run net_kinship_7b_per_sentence_within_seed2 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_7b_per_sentence_within_seed2.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_7b_per_sentence_within_seed2 R hno t

#print axioms correct_kinship_7b_per_sentence_within_seed2
#print axioms no_false_yes_kinship_7b_per_sentence_within_seed2

/-- kinship/qwen2.5-7b/within/seed0: smallest |score| over all cases 0.0013; on the patterns that occur, with no kept bit, 0.1297. -/
def net_kinship_7b_within_seed0 : Net := { tbl := [16748032, 16777065, 15895040, 16226888, 12648448, 16747008, 8388608, 10519040, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 8388608, 16157184, 8388608, 10650112, 0, 8388608, 0, 0, 16777215, 16777215, 16777213, 16777213, 16777214, 16777215, 16772920, 16777085, 8388608, 16157184, 8388608, 10519040, 0, 8388608, 0, 0, 16777215, 16777215, 16777215, 16777213, 16773054, 16777213, 16772912, 16777084, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 14778880, 16751176, 10485760, 16155144, 4194304, 14713344, 0, 2097152, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16747008, 16751176, 10519040, 11967048, 12582912, 16220672, 0, 10519040, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16772904], rd := 236 }

def cert_data_kinship_7b_within_seed0 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, true⟩, Rl := ⟨false, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_within_seed0 : checkSound net_kinship_7b_within_seed0 = true := by decide

theorem cert_kinship_7b_within_seed0 : checkCert net_kinship_7b_within_seed0 within cert_data_kinship_7b_within_seed0 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_7b_within_seed0 : net_kinship_7b_within_seed0.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-7b/within/seed0 is right on every well-formed instance of every size. -/
theorem correct_kinship_7b_within_seed0 {α : Type} (I : Inst α) (R : Run net_kinship_7b_within_seed0 I.pat) :
    (∀ t, net_kinship_7b_within_seed0.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_7b_within_seed0.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_7b_within_seed0 cert_kinship_7b_within_seed0 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_7b_within_seed0 {α : Type} (I : Inst α) (R : Run net_kinship_7b_within_seed0 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_7b_within_seed0.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_7b_within_seed0 R hno t

#print axioms correct_kinship_7b_within_seed0
#print axioms no_false_yes_kinship_7b_within_seed0

/-- kinship/qwen2.5-7b/within/seed1: smallest |score| over all cases 0.0016; on the patterns that occur, with no kept bit, 0.3002. -/
def net_kinship_7b_within_seed1 : Net := { tbl := [16748032, 16177960, 6324224, 2392576, 8388608, 8519680, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16773052, 16768876, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 0, 0, 0, 0, 0, 0, 0, 16777215, 16777213, 16777148, 16768876, 16510388, 16764716, 14353408, 15894528, 0, 2097152, 0, 0, 0, 0, 0, 0, 16777215, 16777213, 16777148, 16768876, 16510388, 16764732, 14371840, 15911200, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16764456, 16178024, 6324224, 10781184, 8388608, 10616832, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777214, 16777213, 16222720, 11983368, 2129920, 2392584, 0, 8388608, 0, 0, 16777215, 16777215, 16777215, 16777213, 16777215, 16777213, 16777150, 16768877, 16745984], rd := 248 }

def cert_data_kinship_7b_within_seed1 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, false⟩, Rl := ⟨false, false, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_within_seed1 : checkSound net_kinship_7b_within_seed1 = true := by decide

theorem cert_kinship_7b_within_seed1 : checkCert net_kinship_7b_within_seed1 within cert_data_kinship_7b_within_seed1 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_7b_within_seed1 : net_kinship_7b_within_seed1.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-7b/within/seed1 is right on every well-formed instance of every size. -/
theorem correct_kinship_7b_within_seed1 {α : Type} (I : Inst α) (R : Run net_kinship_7b_within_seed1 I.pat) :
    (∀ t, net_kinship_7b_within_seed1.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_7b_within_seed1.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_7b_within_seed1 cert_kinship_7b_within_seed1 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_7b_within_seed1 {α : Type} (I : Inst α) (R : Run net_kinship_7b_within_seed1 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_7b_within_seed1.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_7b_within_seed1 R hno t

#print axioms correct_kinship_7b_within_seed1
#print axioms no_false_yes_kinship_7b_within_seed1

/-- kinship/qwen2.5-7b/within/seed2: smallest |score| over all cases 0.0047; on the patterns that occur, with no kept bit, 0.0864. -/
def net_kinship_7b_within_seed2 : Net := { tbl := [16745984, 16777215, 16678912, 16777212, 15728640, 16777208, 8388608, 16764704, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 15728640, 16777208, 15728640, 16777128, 8388608, 16764704, 8388608, 16402720, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777213, 16777215, 15859712, 16777212, 13631488, 16777148, 8388608, 16764704, 8388608, 16468256, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16154624, 16777208, 15859712, 16777208, 8388608, 16764704, 8388608, 16764704, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777213, 16777215, 16748032, 16777215, 16449536, 16777212, 13631488, 16777212, 8388608, 16764708, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777176], rd := 238 }

def cert_data_kinship_7b_within_seed2 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, true, true⟩, Rl := ⟨true, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_within_seed2 : checkSound net_kinship_7b_within_seed2 = true := by decide

theorem cert_kinship_7b_within_seed2 : checkCert net_kinship_7b_within_seed2 within cert_data_kinship_7b_within_seed2 = true := by decide

/-- More kept bits never stop a test of this network (all 113 patterns). -/
theorem mono_kinship_7b_within_seed2 : net_kinship_7b_within_seed2.Mono := checkMono_spec (by decide)

/-- The trained network kinship/qwen2.5-7b/within/seed2 is right on every well-formed instance of every size. -/
theorem correct_kinship_7b_within_seed2 {α : Type} (I : Inst α) (R : Run net_kinship_7b_within_seed2 I.pat) :
    (∀ t, net_kinship_7b_within_seed2.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net_kinship_7b_within_seed2.read (R.L t' I.x I.y) = true) :=
  within_correct I sound_kinship_7b_within_seed2 cert_kinship_7b_within_seed2 R

/-- The same network says no at every loop whenever Y is not linked from X (the dual form). -/
theorem no_false_yes_kinship_7b_within_seed2 {α : Type} (I : Inst α) (R : Run net_kinship_7b_within_seed2 I.pat) (hno : ¬ Reach I.link I.x I.y) (t : Nat) :
    net_kinship_7b_within_seed2.read (R.L t I.x I.y) = false := within_no_false_yes I sound_kinship_7b_within_seed2 R hno t

#print axioms correct_kinship_7b_within_seed2
#print axioms no_false_yes_kinship_7b_within_seed2

/-! ## Layout `partner`: table-level checks only (the pattern lemmas of this layout are measured, not proved) -/

/-- kinship/qwen2.5-7b/per_sentence_partner/seed0. -/
def net_kinship_7b_per_sentence_partner_seed0 : Net := { tbl := [16747008, 16777065, 2130432, 10785352, 13830144, 16502272, 0, 10485760, 16777215, 16777215, 16777211, 16777215, 16777215, 16777215, 16777211, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 10912256, 0, 2130432, 0, 8388608, 0, 0, 16777215, 16777215, 16161352, 16243305, 16777214, 16777215, 16155136, 16767561, 8388608, 10650112, 0, 2097152, 0, 8388608, 0, 0, 16777215, 16777215, 16226888, 16768585, 16773054, 16777215, 14909440, 16752200, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 10781184, 0, 2130432, 0, 0, 0, 0, 16777211, 16777211, 10785353, 16226889, 16777210, 16777211, 10781184, 15176265, 16772664, 16777209, 10781184, 11965001, 16460816, 16773048, 0, 10519040, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16772648], rd := 236 }

def cert_data_kinship_7b_per_sentence_partner_seed0 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, false, false⟩, Rl := ⟨false, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_per_sentence_partner_seed0 : checkSoundOn partnerOcc partnerLink partnerId net_kinship_7b_per_sentence_partner_seed0 = true := by decide

theorem cert_kinship_7b_per_sentence_partner_seed0 : checkCert net_kinship_7b_per_sentence_partner_seed0 partner cert_data_kinship_7b_per_sentence_partner_seed0 = true := by decide

theorem mono_kinship_7b_per_sentence_partner_seed0 : net_kinship_7b_per_sentence_partner_seed0.Mono := checkMono_spec (by decide)

#print axioms cert_kinship_7b_per_sentence_partner_seed0

/-- kinship/qwen2.5-7b/per_sentence_partner/seed1. -/
def net_kinship_7b_per_sentence_partner_seed1 : Net := { tbl := [16748032, 16243208, 2129920, 2392576, 12582912, 12582912, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777214, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 0, 0, 0, 0, 0, 0, 0, 16777215, 16777215, 16777215, 16777213, 16777150, 16777214, 16510096, 16772920, 0, 0, 0, 0, 0, 0, 0, 0, 16777215, 16777215, 16777213, 16777213, 16777150, 16777215, 16510000, 16772920, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 11716608, 11983400, 0, 2129920, 0, 8388608, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777213, 16777016, 16768873, 6586888, 10785352, 14355456, 16469024, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16220672], rd := 248 }

def cert_data_kinship_7b_per_sentence_partner_seed1 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, false, false⟩, Rl := ⟨false, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_per_sentence_partner_seed1 : checkSoundOn partnerOcc partnerLink partnerId net_kinship_7b_per_sentence_partner_seed1 = true := by decide

theorem cert_kinship_7b_per_sentence_partner_seed1 : checkCert net_kinship_7b_per_sentence_partner_seed1 partner cert_data_kinship_7b_per_sentence_partner_seed1 = true := by decide

theorem mono_kinship_7b_per_sentence_partner_seed1 : net_kinship_7b_per_sentence_partner_seed1.Mono := checkMono_spec (by decide)

#print axioms cert_kinship_7b_per_sentence_partner_seed1

/-- kinship/qwen2.5-7b/partner/seed0. -/
def net_kinship_7b_partner_seed0 : Net := { tbl := [16747008, 16777065, 2130432, 10785352, 13830144, 16502272, 0, 10485760, 16777215, 16777215, 16777211, 16777215, 16777215, 16777215, 16777211, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 10912256, 0, 2130432, 0, 8388608, 0, 0, 16777215, 16777215, 16161352, 16243305, 16777214, 16777215, 16155136, 16767561, 8388608, 10650112, 0, 2097152, 0, 8388608, 0, 0, 16777215, 16777215, 16226888, 16767561, 16773054, 16777215, 14909440, 16751176, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 10781184, 0, 2130432, 0, 0, 0, 0, 16777211, 16777211, 10785353, 16226889, 16777210, 16777211, 10781184, 15176265, 16772664, 16777209, 10781184, 11967049, 16460816, 16773048, 0, 10519040, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16772648], rd := 236 }

def cert_data_kinship_7b_partner_seed0 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, false, false⟩, Rl := ⟨false, true, true⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_partner_seed0 : checkSoundOn partnerOcc partnerLink partnerId net_kinship_7b_partner_seed0 = true := by decide

theorem cert_kinship_7b_partner_seed0 : checkCert net_kinship_7b_partner_seed0 partner cert_data_kinship_7b_partner_seed0 = true := by decide

theorem mono_kinship_7b_partner_seed0 : net_kinship_7b_partner_seed0.Mono := checkMono_spec (by decide)

#print axioms cert_kinship_7b_partner_seed0

/-- kinship/qwen2.5-7b/partner/seed1. -/
def net_kinship_7b_partner_seed1 : Net := { tbl := [16748032, 16243208, 2129920, 2392576, 4194304, 12582912, 0, 0, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 0, 0, 0, 0, 0, 0, 0, 0, 16777215, 16777215, 16777213, 16777213, 16777150, 16777214, 16510096, 16772912, 0, 0, 0, 0, 0, 0, 0, 0, 16777215, 16777215, 16777213, 16777213, 16777150, 16777215, 16510000, 16773048, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 10616832, 11960832, 0, 2097152, 0, 0, 0, 0, 16777215, 16777215, 16777215, 16777213, 16777215, 16777215, 16777212, 16777213, 16777212, 16777084, 15045128, 15045192, 14380080, 16477488, 4194304, 4194304, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16777215, 16744960], rd := 248 }

def cert_data_kinship_7b_partner_seed1 : Cert := { G := ⟨true, true, true⟩, GX := ⟨true, true, true⟩, GY := ⟨true, true, true⟩, Rr := ⟨true, false, false⟩, Rl := ⟨false, true, false⟩, Rs := ⟨true, true, true⟩ }

theorem sound_kinship_7b_partner_seed1 : checkSoundOn partnerOcc partnerLink partnerId net_kinship_7b_partner_seed1 = true := by decide

theorem cert_kinship_7b_partner_seed1 : checkCert net_kinship_7b_partner_seed1 partner cert_data_kinship_7b_partner_seed1 = true := by decide

theorem mono_kinship_7b_partner_seed1 : net_kinship_7b_partner_seed1.Mono := checkMono_spec (by decide)

#print axioms cert_kinship_7b_partner_seed1

end Organ
