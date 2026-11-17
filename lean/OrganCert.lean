/-
OrganCert.lean  (Lean 4 core, no Mathlib).  15 Nov 2026.

A trained cold pair block, proved right at every input size.

The block (organ.py, `kinship_run`): every ordered pair (a, b) of items holds three kept bits. One loop:
bit c of (a, b) turns on if some item k passes test c at (a, k) and test c at (k, b); bits stay on. Test c at a pair is a
function of the pair's input pattern and of the pair's own three kept bits. The answer is read from the kept bits of the
pair (X, Y). Patterns take finitely many values and kept bits 8, so the trained network is a finite table (`Net`).

What is proved here, for every item type α (any number of items), every instance, every loop count:
  * `Run.sound`       : if no test passes on a non-link with no kept bit, a kept bit at (a, b) implies a chain of links a → b.
  * `chain_complete`  : if the table passes the finite check `checkCert`, a chain of ℓ links whose pairs carry the expected
                        patterns is answered yes by loop ⌈log2 (ℓ - 2)⌉ + 2, and at every later loop.
  * `within_correct`  : for the kinship layout `within`, under four plain well-formedness conditions, the read-out
                        says yes iff X is linked to Y by a chain of "same person" and "parent → child" links.
The tables of the trained networks and the two `decide` checks for each are in OrganTables.lean (generated).
What is NOT proved in Lean: that the table equals the float network (it is the network's forward pass on its finite
per-pair domain, evaluated exhaustively on every case, signs equal in float32 and float64).
-/
set_option autoImplicit false
set_option linter.deprecated false
open Classical

namespace Organ

/-! ## Kept bits, channels, the table -/

inductive Ch where
  | c0 | c1 | c2
deriving DecidableEq, Repr

def Ch.val : Ch → Nat
  | .c0 => 0
  | .c1 => 1
  | .c2 => 2

def Ch.all (f : Ch → Bool) : Bool := f .c0 && f .c1 && f .c2

theorem Ch.all_spec {f : Ch → Bool} (h : Ch.all f = true) (c : Ch) : f c = true := by
  simp only [Ch.all, Bool.and_eq_true] at h
  cases c
  · exact h.1.1
  · exact h.1.2
  · exact h.2

structure Bits where
  b0 : Bool
  b1 : Bool
  b2 : Bool
deriving DecidableEq, Repr

namespace Bits

def zero : Bits := ⟨false, false, false⟩

def get (s : Bits) : Ch → Bool
  | .c0 => s.b0
  | .c1 => s.b1
  | .c2 => s.b2

/-- The index used by the extraction script: 4 * L0 + 2 * L1 + L2. -/
def idx (s : Bits) : Nat := 4 * s.b0.toNat + 2 * s.b1.toNat + s.b2.toNat

/-- Every bit of `g` is on in `s`. -/
def le (g s : Bits) : Bool := (!g.b0 || s.b0) && (!g.b1 || s.b1) && (!g.b2 || s.b2)

def union (g h : Bits) : Bits := ⟨g.b0 || h.b0, g.b1 || h.b1, g.b2 || h.b2⟩

def all (f : Bits → Bool) : Bool :=
  f ⟨false, false, false⟩ && f ⟨false, false, true⟩ && f ⟨false, true, false⟩ && f ⟨false, true, true⟩ &&
  f ⟨true, false, false⟩ && f ⟨true, false, true⟩ && f ⟨true, true, false⟩ && f ⟨true, true, true⟩

theorem all_spec {f : Bits → Bool} (h : all f = true) (s : Bits) : f s = true := by
  simp only [all, Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨h0, h1⟩, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩ := h
  obtain ⟨b0, b1, b2⟩ := s
  cases b0 <;> cases b1 <;> cases b2 <;> assumption

theorem le_of_get {g s : Bits} (h : ∀ c, g.get c = true → s.get c = true) : le g s = true := by
  obtain ⟨g0, g1, g2⟩ := g
  obtain ⟨s0, s1, s2⟩ := s
  have h0 := h .c0
  have h1 := h .c1
  have h2 := h .c2
  simp only [get] at h0 h1 h2
  cases g0 <;> cases g1 <;> cases g2 <;> simp_all [le]

theorem get_of_le {g s : Bits} (h : le g s = true) (c : Ch) (hc : g.get c = true) : s.get c = true := by
  obtain ⟨g0, g1, g2⟩ := g
  obtain ⟨s0, s1, s2⟩ := s
  cases c <;> simp_all [le, get]

theorem union_get {g h : Bits} (c : Ch) : (g.union h).get c = true ↔ (g.get c = true ∨ h.get c = true) := by
  obtain ⟨g0, g1, g2⟩ := g
  obtain ⟨h0, h1, h2⟩ := h
  cases c <;> simp [union, get]

theorem exists_get_of_ne_zero {s : Bits} (h : s ≠ zero) : ∃ c, s.get c = true := by
  obtain ⟨s0, s1, s2⟩ := s
  cases s0
  · cases s1
    · cases s2
      · exact absurd rfl h
      · exact ⟨.c2, rfl⟩
    · exact ⟨.c1, rfl⟩
  · exact ⟨.c0, rfl⟩

end Bits

/-- A trained block as a finite table. `tbl[π]` has bit `3 * (state index) + channel` set iff test `channel` passes on
pattern `π` with that kept state; `rd` has bit `state index` set iff the read-out says yes on that kept state. -/
structure Net where
  tbl : List Nat
  rd : Nat

def Net.test (net : Net) (c : Ch) (p : Nat) (s : Bits) : Bool := Nat.testBit (net.tbl.getD p 0) (3 * s.idx + c.val)

def Net.read (net : Net) (s : Bits) : Bool := Nat.testBit net.rd s.idx

/-- Test `c` passes on pattern `p` whatever the kept bits. -/
def Net.always (net : Net) (c : Ch) (p : Nat) : Bool := Bits.all fun s => net.test c p s

/-- Test `c` passes on pattern `p` whenever the bits of `g` are on. -/
def Net.above (net : Net) (c : Ch) (p : Nat) (g : Bits) : Bool := Bits.all fun s => !(g.le s) || net.test c p s

def Net.alwaysOn (net : Net) (c : Ch) (ps : List Nat) : Bool := ps.all fun p => net.always c p

theorem Net.always_spec {net : Net} {c : Ch} {p : Nat} (h : net.always c p = true) (s : Bits) : net.test c p s = true :=
  Bits.all_spec h s

theorem Net.above_spec {net : Net} {c : Ch} {p : Nat} {g : Bits} (h : net.above c p g = true) (s : Bits)
    (hs : g.le s = true) : net.test c p s = true := by
  have := Bits.all_spec h s
  simp only [hs, Bool.not_true, Bool.false_or] at this
  exact this

theorem Net.alwaysOn_spec {net : Net} {c : Ch} {ps : List Nat} (h : net.alwaysOn c ps = true) {p : Nat} (hp : p ∈ ps)
    (s : Bits) : net.test c p s = true :=
  Net.always_spec (List.all_eq_true.mp h p hp) s

/-! ## A run of the block -/

/-- The states of the block on an instance: `L t a b` are the kept bits of the pair (a, b) after `t` loops.
`step` is the forward pass of `organ.kinship_run`: `L2 = max(L, matmul(x > 0, x > 0) > 0)`. -/
structure Run (net : Net) {α : Type} (pat : α → α → Nat) where
  L : Nat → α → α → Bits
  init : ∀ a b, L 0 a b = Bits.zero
  step : ∀ t a b c, (L (t + 1) a b).get c = true ↔
    ((L t a b).get c = true ∨
      ∃ k, net.test c (pat a k) (L t a k) = true ∧ net.test c (pat k b) (L t k b) = true)

section generic
variable {net : Net} {α : Type} {pat : α → α → Nat}

theorem Run.mono (R : Run net pat) {t : Nat} {a b : α} {c : Ch} (h : (R.L t a b).get c = true) :
    ∀ t', t ≤ t' → (R.L t' a b).get c = true := by
  intro t' ht
  induction t' with
  | zero =>
    have : t = 0 := by omega
    subst this
    exact h
  | succ n ih =>
    by_cases hn : t ≤ n
    · exact (R.step n a b c).mpr (Or.inl (ih hn))
    · have : t = n + 1 := by omega
      subst this
      exact h

/-! ## Soundness: a kept bit is a chain of links -/

/-- `Reach link a b`: a chain of zero or more links from a to b. -/
inductive Reach (link : α → α → Prop) : α → α → Prop where
  | refl (a : α) : Reach link a a
  | tail {a b c : α} : Reach link a b → link b c → Reach link a c

theorem Reach.single {link : α → α → Prop} {a b : α} (h : link a b) : Reach link a b :=
  Reach.tail (Reach.refl a) h

theorem Reach.trans {link : α → α → Prop} {a b c : α} (h1 : Reach link a b) (h2 : Reach link b c) : Reach link a c := by
  induction h2 with
  | refl => exact h1
  | tail _ hl ih => exact Reach.tail ih hl

/-- If, with no kept bit, a test passes only on a link or on a pair (a, a), then every kept bit that is on at (a, b),
at any loop, comes with a chain of links from a to b. No assumption on the size of the instance. -/
theorem Run.sound (R : Run net pat) {link : α → α → Prop}
    (h0 : ∀ c a b, net.test c (pat a b) Bits.zero = true → link a b ∨ a = b) :
    ∀ t a b c, (R.L t a b).get c = true → Reach link a b := by
  intro t
  induction t with
  | zero =>
    intro a b c h
    rw [R.init] at h
    cases c <;> simp [Bits.zero, Bits.get] at h
  | succ n ih =>
    intro a b c h
    have leg : ∀ a k, net.test c (pat a k) (R.L n a k) = true → Reach link a k := by
      intro a k hk
      by_cases hz : R.L n a k = Bits.zero
      · rw [hz] at hk
        rcases h0 c a k hk with hl | he
        · exact Reach.single hl
        · subst he
          exact Reach.refl a
      · obtain ⟨c', hc'⟩ := Bits.exists_get_of_ne_zero hz
        exact ih a k c' hc'
    rcases (R.step n a b c).mp h with h1 | ⟨k, hk1, hk2⟩
    · exact ih a b c h1
    · exact Reach.trans (leg a k hk1) (leg k b hk2)

/-- Soundness in its most general form: an invariant with one meaning per channel. Give every channel c a relation
`M c` on items ("what bit c claims"). If one loop respects the meanings (whenever the kept bits of two legs mean what
they claim and test c passes on both, the joined pair satisfies `M c`), then every kept bit that is ever on means what
it claims, on an instance of any size. `Run.sound` is the case where every channel claims "a chain of links".
With three meanings (b is reachable from a; a is reachable from b; both) this is the positive mask for "same
component". -/
theorem Run.invariant (R : Run net pat) (M : Ch → α → α → Prop)
    (hstep : ∀ (c : Ch) (a k b : α) (s1 s2 : Bits),
      (∀ c', s1.get c' = true → M c' a k) → (∀ c', s2.get c' = true → M c' k b) →
      net.test c (pat a k) s1 = true → net.test c (pat k b) s2 = true → M c a b) :
    ∀ t a b c, (R.L t a b).get c = true → M c a b := by
  intro t
  induction t with
  | zero =>
    intro a b c h
    rw [R.init] at h
    cases c <;> simp [Bits.zero, Bits.get] at h
  | succ n ih =>
    intro a b c h
    rcases (R.step n a b c).mp h with h1 | ⟨k, hk1, hk2⟩
    · exact ih a b c h1
    · exact hstep c a k b _ _ (fun c' hc' => ih a k c' hc') (fun c' hc' => ih k b c' hc') hk1 hk2

/-! ## Soundness in dual form: a simulation into a small abstract instance

A dual form, with one correction to its first statement: the
input patterns of a real pair are not kept when items are merged, so the dual does not carry input facts. It carries,
for each of its pairs, the kept bits a real pair mapped there may ever hold, and the patterns such a pair may carry. -/

/-- A small abstract instance (a "dual", or template). `B p q`: the kept bits a pair of real items mapped to the dual
pair (p, q) may ever carry. `allowed p q π`: such a pair may carry the input pattern π. -/
structure Dual (δ : Type) where
  B : δ → δ → Bits
  allowed : δ → δ → Nat → Prop

/-- The simulation table: the dual is closed under one loop. If two legs with allowed patterns fire on kept states
inside the dual's, the dual's pair carries the bit. Every smaller kept state is quantified over, so nothing is asked
of the signs of the weights. -/
def Dual.Closed (net : Net) {δ : Type} (D : Dual δ) : Prop :=
  ∀ (c : Ch) (p m q : δ) (π1 π2 : Nat) (s1 s2 : Bits),
    D.allowed p m π1 → D.allowed m q π2 → Bits.le s1 (D.B p m) = true → Bits.le s2 (D.B m q) = true →
    net.test c π1 s1 = true → net.test c π2 s2 = true → (D.B p q).get c = true

theorem Bits.zero_le (s : Bits) : Bits.le Bits.zero s = true := by
  obtain ⟨s0, s1, s2⟩ := s
  simp [Bits.le, Bits.zero]

theorem Bits.eq_zero_of_le_zero {s : Bits} (h : Bits.le s Bits.zero = true) : s = Bits.zero := by
  obtain ⟨s0, s1, s2⟩ := s
  cases s0 <;> cases s1 <;> cases s2 <;> simp_all [Bits.le, Bits.zero]

/-- Simulation. If every pair of real items carries a pattern its dual pair allows, then at every loop the kept bits
of every real pair stay inside those of its dual pair. Any number of items. -/
theorem Run.simulate (R : Run net pat) {δ : Type} (D : Dual δ) (hD : D.Closed net) (h : α → δ)
    (hpat : ∀ a b, D.allowed (h a) (h b) (pat a b)) :
    ∀ t a b, Bits.le (R.L t a b) (D.B (h a) (h b)) = true := by
  intro t
  induction t with
  | zero =>
    intro a b
    rw [R.init]
    exact Bits.zero_le _
  | succ n ih =>
    intro a b
    apply Bits.le_of_get
    intro c hc
    rcases (R.step n a b c).mp hc with h1 | ⟨k, hk1, hk2⟩
    · exact Bits.get_of_le (ih a b) c h1
    · exact hD c (h a) (h k) (h b) _ _ _ _ (hpat a k) (hpat k b) (ih a k) (ih k b) hk1 hk2

/-- No false yes, dual form: if the read-out says no on every kept state the dual allows at the image of the question
pair, it says no on the real instance at every loop. -/
theorem Run.no_false_yes (R : Run net pat) {δ : Type} (D : Dual δ) (hD : D.Closed net) (h : α → δ)
    (hpat : ∀ a b, D.allowed (h a) (h b) (pat a b)) (x y : α)
    (hread : ∀ s, Bits.le s (D.B (h x) (h y)) = true → net.read s = false) (t : Nat) :
    net.read (R.L t x y) = false :=
  hread _ (R.simulate D hD h hpat t x y)

/-! ## Lower bounds that do not assume one mechanism

The trained networks reach the same closure in different ways (the certificates beside these files): some keep nothing
on a single step, some raise a bit through an item's pair with itself, some attach the question at one end only, one
walks the chain from Y's end. The three lemmas below are the class lemmas for "never a false no" that allow all of
these. They need the tests to be monotone in the kept bits (`Net.Mono`), which is a finite check on a table. -/

/-- More kept bits never stop a test. -/
def Net.Mono (net : Net) : Prop :=
  ∀ (c : Ch) (p : Nat) (s s' : Bits), Bits.le s s' = true → net.test c p s = true → net.test c p s' = true

/-- The finite check that a table is monotone in the kept bits. -/
def checkMono (net : Net) : Bool :=
  (List.range net.tbl.length).all fun p => Ch.all fun c => Bits.all fun s => Bits.all fun s' =>
    !(Bits.le s s') || !(net.test c p s) || net.test c p s'

theorem checkMono_spec {net : Net} (h : checkMono net = true) : net.Mono := by
  intro c p s s' hle ht
  by_cases hp : p < net.tbl.length
  · have h1 := List.all_eq_true.mp h p (List.mem_range.mpr hp)
    have h2 := Bits.all_spec (Bits.all_spec (Ch.all_spec h1 c) s) s'
    rw [hle, ht] at h2
    simpa using h2
  · exfalso
    have hnone : net.tbl[p]? = none := List.getElem?_eq_none (by omega)
    have : net.test c p s = false := by
      simp [Net.test, List.getD, hnone]
    rw [this] at ht
    exact Bool.noConfusion ht

/-- Any claimed family of kept bits that justifies itself loop by loop is really there. `LB t a b` is a claim about the
bits of (a, b) after t loops. If nothing is claimed at loop 0, and every bit claimed at loop t + 1 was claimed at loop
t or is derived through some item k from bits claimed at loop t, then the claim is below the true state at every loop.
The middle item k may be a or b themselves. -/
theorem Run.lower_bound (R : Run net pat) (hmono : net.Mono) (LB : Nat → α → α → Bits)
    (h0 : ∀ a b, LB 0 a b = Bits.zero)
    (hstep : ∀ t a b c, (LB (t + 1) a b).get c = true →
      (LB t a b).get c = true ∨
        ∃ k, net.test c (pat a k) (LB t a k) = true ∧ net.test c (pat k b) (LB t k b) = true) :
    ∀ t a b, Bits.le (LB t a b) (R.L t a b) = true := by
  intro t
  induction t with
  | zero =>
    intro a b
    rw [h0]
    exact Bits.zero_le _
  | succ n ih =>
    intro a b
    apply Bits.le_of_get
    intro c hc
    rcases hstep n a b c hc with h1 | ⟨k, hk1, hk2⟩
    · exact (R.step n a b c).mpr (Or.inl (Bits.get_of_le (ih a b) c h1))
    · exact (R.step n a b c).mpr (Or.inr ⟨k, hmono c _ _ _ (ih a k) hk1, hmono c _ _ _ (ih k b) hk2⟩)

/-- A sub-context has no more kept bits than the context. If the items of a context β are mapped into a context α with
the same pattern on every pair, then at every loop the kept bits of a pair of β are among those of its image. So a
bare chain inside a larger family bounds the family from below. -/
theorem Run.subcontext {β : Type} {pat' : β → β → Nat} (R : Run net pat) (R' : Run net pat') (hmono : net.Mono)
    (f : β → α) (hf : ∀ a b, pat (f a) (f b) = pat' a b) :
    ∀ t a b, Bits.le (R'.L t a b) (R.L t (f a) (f b)) = true := by
  intro t
  induction t with
  | zero =>
    intro a b
    rw [R'.init]
    exact Bits.zero_le _
  | succ n ih =>
    intro a b
    apply Bits.le_of_get
    intro c hc
    rcases (R'.step n a b c).mp hc with h1 | ⟨k, hk1, hk2⟩
    · exact (R.step n (f a) (f b) c).mpr (Or.inl (Bits.get_of_le (ih a b) c h1))
    · refine (R.step n (f a) (f b) c).mpr (Or.inr ⟨f k, ?_, ?_⟩)
      · rw [hf]; exact hmono c _ _ _ (ih a k) hk1
      · rw [hf]; exact hmono c _ _ _ (ih k b) hk2

/-- M, 2M, 4M, ... : the longest stretch covered after t more loops. -/
def dbl (M : Nat) : Nat → Nat
  | 0 => M
  | t + 1 => 2 * dbl M t

theorem le_dbl (M t : Nat) : M ≤ dbl M t := by
  induction t with
  | zero => exact Nat.le_refl M
  | succ n ih => simp only [dbl]; omega

/-- Kept bits stay on: a set of bits present at loop t is present at every later loop. -/
theorem Run.le_mono (R : Run net pat) {S : Bits} {t : Nat} {a b : α} (h : Bits.le S (R.L t a b) = true) :
    ∀ t', t ≤ t' → Bits.le S (R.L t' a b) = true := by
  intro t' ht
  apply Bits.le_of_get
  intro c hc
  exact R.mono (Bits.get_of_le h c hc) t' ht

/-- Induction on distance from a base band (the every-length step of the certificates' argument). On a stretch v lo, ...,
v hi of a chain whose non-neighbour pairs all carry the pattern `pLong`, let S be a set of bits that is re-derived when
both halves of a pair carry it. If S is on every pair between m and M steps apart by loop τ, with M + 1 ≥ 2m, then it
is on every pair up to M·2^t steps apart by loop τ + t. Nothing is assumed about single steps or about how the base
band was reached. -/
theorem band_double (R : Run net pat) {v : Nat → α} {S : Bits} {pLong lo hi m M tau : Nat}
    (hm : 2 ≤ m) (hM : 2 * m ≤ M + 1)
    (hpat : ∀ i j, lo ≤ i → i + 2 ≤ j → j ≤ hi → pat (v i) (v j) = pLong)
    (hS : ∀ c, S.get c = true → ∀ s, Bits.le S s = true → net.test c pLong s = true)
    (base : ∀ i j, lo ≤ i → j ≤ hi → i + m ≤ j → j ≤ i + M → Bits.le S (R.L tau (v i) (v j)) = true) :
    ∀ t i j, lo ≤ i → j ≤ hi → i + m ≤ j → j ≤ i + dbl M t → Bits.le S (R.L (tau + t) (v i) (v j)) = true := by
  intro t
  induction t with
  | zero =>
    intro i j hi' hj hmj hMj
    exact base i j hi' hj hmj hMj
  | succ n ih =>
    intro i j hi' hj hmj hMj
    have hle := le_dbl M n
    simp only [dbl] at hMj
    by_cases hsmall : j ≤ i + dbl M n
    · exact R.le_mono (ih i j hi' hj hmj hsmall) (tau + (n + 1)) (by omega)
    · apply Bits.le_of_get
      intro c hc
      have e : tau + (n + 1) = tau + n + 1 := by omega
      rw [e]
      refine (R.step (tau + n) (v i) (v j) c).mpr (Or.inr ⟨v (i + (j - i) / 2), ?_, ?_⟩)
      · rw [hpat i (i + (j - i) / 2) hi' (by omega) (by omega)]
        exact hS c hc _ (ih i (i + (j - i) / 2) hi' (by omega) (by omega) (by omega))
      · rw [hpat (i + (j - i) / 2) j (by omega) (by omega) hj]
        exact hS c hc _ (ih (i + (j - i) / 2) j (by omega) hj (by omega) (by omega))

theorem dbl_mono {M M' : Nat} (h : M ≤ M') (t : Nat) : dbl M t ≤ dbl M' t := by
  induction t with
  | zero => exact h
  | succ n ih => simp only [dbl]; omega

/-- One split (the step rule read as a lemma). If test c fires at (a, k) on any state that holds S1 and at (k, b) on any
state that holds S2, and those sets are on after loop t, then bit c is on at (a, b) after loop t + 1. This one lemma
covers: X or Y attached through the first or last single step; a bit raised through an item's pair with itself
(k = a or k = b); the question's pair split in the middle, after the first step, or before the last. -/
theorem Run.join (R : Run net pat) {a k b : α} {c : Ch} {S1 S2 : Bits} {t : Nat}
    (h1 : ∀ s, Bits.le S1 s = true → net.test c (pat a k) s = true)
    (h2 : ∀ s, Bits.le S2 s = true → net.test c (pat k b) s = true)
    (l1 : Bits.le S1 (R.L t a k) = true) (l2 : Bits.le S2 (R.L t k b) = true) :
    (R.L (t + 1) a b).get c = true :=
  (R.step t a b c).mpr (Or.inr ⟨k, h1 _ l1, h2 _ l2⟩)

/-- The question's pair is one more split: if every bit of a set Q is derived by joining S1 at (a, k) with S2 at (k, b),
and the read-out says yes on any state that holds Q, the answer is yes one loop later. -/
theorem Run.join_read (R : Run net pat) {a k b : α} {S1 S2 Q : Bits} {t : Nat}
    (hq : ∀ c, Q.get c = true → (∀ s, Bits.le S1 s = true → net.test c (pat a k) s = true) ∧
      (∀ s, Bits.le S2 s = true → net.test c (pat k b) s = true))
    (hread : ∀ s, Bits.le Q s = true → net.read s = true)
    (l1 : Bits.le S1 (R.L t a k) = true) (l2 : Bits.le S2 (R.L t k b) = true) :
    net.read (R.L (t + 1) a b) = true :=
  hread _ (Bits.le_of_get (fun c hc => R.join (hq c hc).1 (hq c hc).2 l1 l2))

/-- Attaching the left end by halving. An item a (X in our layouts) stands before the stretch v lo, ..., v hi; its
pairs (a, v j) carry the pattern `pA` from j = lo + mA on. Let SA be a set of bits re-derived at (a, v j) from SA on
(a, middle) and the inner set S on (middle, v j). If SA is on the pairs (a, v j) with j - lo between mA and MA by loop
τ, then it is on every pair with j - lo up to MA·2^t by loop τ + t. `inner` is the conclusion of `band_double`. -/
theorem anchor_left_double (R : Run net pat) {v : Nat → α} {a : α} {S SA : Bits} {pLong pA lo hi m M mA MA tau : Nat}
    (hm : 2 ≤ m) (hmA : m ≤ mA) (hMA : 2 * mA ≤ MA + 1) (hMM : MA ≤ M)
    (hpat : ∀ i j, lo ≤ i → i + 2 ≤ j → j ≤ hi → pat (v i) (v j) = pLong)
    (hpatA : ∀ j, lo + mA ≤ j → j ≤ hi → pat a (v j) = pA)
    (hSA : ∀ c, SA.get c = true → (∀ s, Bits.le SA s = true → net.test c pA s = true) ∧
      (∀ s, Bits.le S s = true → net.test c pLong s = true))
    (inner : ∀ t i j, lo ≤ i → j ≤ hi → i + m ≤ j → j ≤ i + dbl M t → Bits.le S (R.L (tau + t) (v i) (v j)) = true)
    (base : ∀ j, lo + mA ≤ j → j ≤ hi → j ≤ lo + MA → Bits.le SA (R.L tau a (v j)) = true) :
    ∀ t j, lo + mA ≤ j → j ≤ hi → j ≤ lo + dbl MA t → Bits.le SA (R.L (tau + t) a (v j)) = true := by
  intro t
  induction t with
  | zero =>
    intro j h1 h2 h3
    exact base j h1 h2 h3
  | succ n ih =>
    intro j h1 h2 h3
    have hle := le_dbl MA n
    have hmono := dbl_mono hMM n
    simp only [dbl] at h3
    by_cases hsmall : j ≤ lo + dbl MA n
    · exact R.le_mono (ih j h1 h2 hsmall) (tau + (n + 1)) (by omega)
    · apply Bits.le_of_get
      intro c hc
      have e : tau + (n + 1) = tau + n + 1 := by omega
      rw [e]
      refine (R.step (tau + n) a (v j) c).mpr (Or.inr ⟨v (lo + (j - lo) / 2), ?_, ?_⟩)
      · rw [hpatA (lo + (j - lo) / 2) (by omega) (by omega)]
        exact (hSA c hc).1 _ (ih (lo + (j - lo) / 2) (by omega) (by omega) (by omega))
      · rw [hpat (lo + (j - lo) / 2) j (by omega) (by omega) h2]
        exact (hSA c hc).2 _ (inner n (lo + (j - lo) / 2) j (by omega) h2 (by omega) (by omega))

/-- Attaching the right end by halving: the mirror image, for an item b (Y) that stands after the stretch. This is
also the shape of a block that works from Y's end. -/
theorem anchor_right_double (R : Run net pat) {v : Nat → α} {b : α} {S SB : Bits} {pLong pB lo hi m M mB MB tau : Nat}
    (hm : 2 ≤ m) (hmB : m ≤ mB) (hMB : 2 * mB ≤ MB + 1) (hMM : MB ≤ M)
    (hpat : ∀ i j, lo ≤ i → i + 2 ≤ j → j ≤ hi → pat (v i) (v j) = pLong)
    (hpatB : ∀ i, lo ≤ i → i + mB ≤ hi → pat (v i) b = pB)
    (hSB : ∀ c, SB.get c = true → (∀ s, Bits.le S s = true → net.test c pLong s = true) ∧
      (∀ s, Bits.le SB s = true → net.test c pB s = true))
    (inner : ∀ t i j, lo ≤ i → j ≤ hi → i + m ≤ j → j ≤ i + dbl M t → Bits.le S (R.L (tau + t) (v i) (v j)) = true)
    (base : ∀ i, lo ≤ i → i + mB ≤ hi → hi ≤ i + MB → Bits.le SB (R.L tau (v i) b) = true) :
    ∀ t i, lo ≤ i → i + mB ≤ hi → hi ≤ i + dbl MB t → Bits.le SB (R.L (tau + t) (v i) b) = true := by
  intro t
  induction t with
  | zero =>
    intro i h1 h2 h3
    exact base i h1 h2 h3
  | succ n ih =>
    intro i h1 h2 h3
    have hle := le_dbl MB n
    have hmono := dbl_mono hMM n
    simp only [dbl] at h3
    by_cases hsmall : hi ≤ i + dbl MB n
    · exact R.le_mono (ih i h1 h2 hsmall) (tau + (n + 1)) (by omega)
    · apply Bits.le_of_get
      intro c hc
      have e : tau + (n + 1) = tau + n + 1 := by omega
      rw [e]
      refine (R.step (tau + n) (v i) b c).mpr (Or.inr ⟨v (hi - (hi - i) / 2), ?_, ?_⟩)
      · rw [hpat i (hi - (hi - i) / 2) h1 (by omega) (by omega)]
        exact (hSB c hc).1 _ (inner n i (hi - (hi - i) / 2) h1 (by omega) (by omega) (by omega))
      · rw [hpatB (hi - (hi - i) / 2) (by omega) (by omega)]
        exact (hSB c hc).2 _ (ih (hi - (hi - i) / 2) (by omega) (by omega) (by omega))

/-! ## Completeness on a chain: doubling, then two caps -/

/-- Which input patterns play which role on a shortest chain of links v 0, ..., v ℓ from X to Y. -/
structure Shape where
  first : List Nat   -- the pattern of the first link (v 0, v 1)
  last : List Nat    -- the pattern of the last link (v (ℓ-1), v ℓ)
  inner : List Nat   -- the patterns of the other links
  pII : Nat          -- the pattern of two inner items that are not neighbours
  pXI : Nat          -- the pattern of (X, an inner item that is not its neighbour)
  pIY : Nat          -- the pattern of (an inner item that is not its neighbour, Y)

/-- The certificate: sets of channels. `G`: channels that close the inner chain by doubling. `GX`, `GY`: those of `G`
that also pass the first / the last link. `Rr`, `Rl`: channels that reach the pair (X, Y) by a right / a left cap.
`Rs`: channels that pass both the first and the last link (chains of two links). -/
structure Cert where
  G : Bits
  GX : Bits
  GY : Bits
  Rr : Bits
  Rl : Bits
  Rs : Bits

/-- The finite check on the table. -/
def checkCert (net : Net) (S : Shape) (C : Cert) : Bool :=
  Ch.all (fun c => !(C.G.get c) || (net.alwaysOn c S.inner && net.above c S.pII C.G)) &&
  C.GX.le C.G && C.GY.le C.G &&
  Ch.all (fun c => !(C.GX.get c) || net.alwaysOn c S.first) &&
  Ch.all (fun c => !(C.GY.get c) || net.alwaysOn c S.last) &&
  Ch.all (fun c => !(C.Rr.get c) || (net.alwaysOn c S.first && net.above c S.pIY C.GY)) &&
  Ch.all (fun c => !(C.Rl.get c) || (net.above c S.pXI C.GX && net.alwaysOn c S.last)) &&
  Ch.all (fun c => !(C.Rs.get c) || (net.alwaysOn c S.first && net.alwaysOn c S.last)) &&
  Bits.all (fun s => !((C.Rr.union C.Rl).le s) || net.read s) &&
  Bits.all (fun s => !(C.Rs.le s) || net.read s)

/-- What the check means. -/
structure CertOK (net : Net) (S : Shape) (C : Cert) : Prop where
  inner : ∀ c, C.G.get c = true → ∀ p, p ∈ S.inner → ∀ s, net.test c p s = true
  closeII : ∀ c, C.G.get c = true → ∀ s, C.G.le s = true → net.test c S.pII s = true
  gx_le : C.GX.le C.G = true
  gy_le : C.GY.le C.G = true
  gx_first : ∀ c, C.GX.get c = true → ∀ p, p ∈ S.first → ∀ s, net.test c p s = true
  gy_last : ∀ c, C.GY.get c = true → ∀ p, p ∈ S.last → ∀ s, net.test c p s = true
  rr_first : ∀ c, C.Rr.get c = true → ∀ p, p ∈ S.first → ∀ s, net.test c p s = true
  rr_IY : ∀ c, C.Rr.get c = true → ∀ s, C.GY.le s = true → net.test c S.pIY s = true
  rl_XI : ∀ c, C.Rl.get c = true → ∀ s, C.GX.le s = true → net.test c S.pXI s = true
  rl_last : ∀ c, C.Rl.get c = true → ∀ p, p ∈ S.last → ∀ s, net.test c p s = true
  rs_first : ∀ c, C.Rs.get c = true → ∀ p, p ∈ S.first → ∀ s, net.test c p s = true
  rs_last : ∀ c, C.Rs.get c = true → ∀ p, p ∈ S.last → ∀ s, net.test c p s = true
  read_long : ∀ s, (C.Rr.union C.Rl).le s = true → net.read s = true
  read_short : ∀ s, C.Rs.le s = true → net.read s = true

theorem imp_of_or {a b : Bool} (h : (!a || b) = true) (ha : a = true) : b = true := by
  cases a <;> simp_all

theorem checkCert_ok {net : Net} {S : Shape} {C : Cert} (h : checkCert net S C = true) : CertOK net S C := by
  simp only [checkCert, Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨hG, hgx⟩, hgy⟩, hGX⟩, hGY⟩, hRr⟩, hRl⟩, hRs⟩, hread⟩, hshort⟩ := h
  refine ⟨?_, ?_, hgx, hgy, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro c hc p hp s
    have := imp_of_or (Ch.all_spec hG c) hc
    simp only [Bool.and_eq_true] at this
    exact Net.alwaysOn_spec this.1 hp s
  · intro c hc s hs
    have := imp_of_or (Ch.all_spec hG c) hc
    simp only [Bool.and_eq_true] at this
    exact Net.above_spec this.2 s hs
  · intro c hc p hp s
    exact Net.alwaysOn_spec (imp_of_or (Ch.all_spec hGX c) hc) hp s
  · intro c hc p hp s
    exact Net.alwaysOn_spec (imp_of_or (Ch.all_spec hGY c) hc) hp s
  · intro c hc p hp s
    have := imp_of_or (Ch.all_spec hRr c) hc
    simp only [Bool.and_eq_true] at this
    exact Net.alwaysOn_spec this.1 hp s
  · intro c hc s hs
    have := imp_of_or (Ch.all_spec hRr c) hc
    simp only [Bool.and_eq_true] at this
    exact Net.above_spec this.2 s hs
  · intro c hc s hs
    have := imp_of_or (Ch.all_spec hRl c) hc
    simp only [Bool.and_eq_true] at this
    exact Net.above_spec this.1 s hs
  · intro c hc p hp s
    have := imp_of_or (Ch.all_spec hRl c) hc
    simp only [Bool.and_eq_true] at this
    exact Net.alwaysOn_spec this.2 hp s
  · intro c hc p hp s
    have := imp_of_or (Ch.all_spec hRs c) hc
    simp only [Bool.and_eq_true] at this
    exact Net.alwaysOn_spec this.1 hp s
  · intro c hc p hp s
    have := imp_of_or (Ch.all_spec hRs c) hc
    simp only [Bool.and_eq_true] at this
    exact Net.alwaysOn_spec this.2 hp s
  · intro s hs
    exact imp_of_or (Bits.all_spec hread s) hs
  · intro s hs
    exact imp_of_or (Bits.all_spec hshort s) hs

/-- A chain v 0, ..., v ℓ whose pairs carry the patterns of `S`. -/
structure Chain (S : Shape) (pat : α → α → Nat) (v : Nat → α) (l : Nat) : Prop where
  len : 2 ≤ l
  first : pat (v 0) (v 1) ∈ S.first
  last : pat (v (l - 1)) (v l) ∈ S.last
  inner : ∀ i, 1 ≤ i → i + 2 ≤ l → pat (v i) (v (i + 1)) ∈ S.inner
  II : ∀ i j, 1 ≤ i → i + 2 ≤ j → j + 1 ≤ l → pat (v i) (v j) = S.pII
  IY : ∀ i, 1 ≤ i → i + 2 ≤ l → pat (v i) (v l) = S.pIY
  XI : ∀ j, 2 ≤ j → j + 1 ≤ l → pat (v 0) (v j) = S.pXI

section chain
variable {S : Shape} {C : Cert} {v : Nat → α} {l : Nat}

/-- Doubling on the inner chain: after t loops, every two inner items at most 2^t links apart carry the bits of G. -/
theorem inner_double (ok : CertOK net S C) (ch : Chain S pat v l) (R : Run net pat) :
    ∀ t i j, 1 ≤ i → i + 2 ≤ j → j + 1 ≤ l → j - i ≤ 2 ^ t → ∀ c, C.G.get c = true →
      (R.L t (v i) (v j)).get c = true := by
  intro t
  induction t with
  | zero =>
    intro i j _ h2 _ hd
    simp at hd
    omega
  | succ n ih =>
    intro i j hi hij hj hd c hc
    have hpow : 2 ^ (n + 1) = 2 * 2 ^ n := by rw [Nat.pow_succ]; omega
    -- a leg (i', j') of at most 2^n links passes test c at loop n
    have leg : ∀ i' j', 1 ≤ i' → i' + 1 ≤ j' → j' + 1 ≤ l → j' - i' ≤ 2 ^ n →
        net.test c (pat (v i') (v j')) (R.L n (v i') (v j')) = true := by
      intro i' j' hi' hij' hj' hd'
      by_cases hadj : j' = i' + 1
      · subst hadj
        exact ok.inner c hc _ (ch.inner i' hi' (by omega)) _
      · rw [ch.II i' j' hi' (by omega) hj']
        exact ok.closeII c hc _ (Bits.le_of_get (fun c' hc' => ih i' j' hi' (by omega) hj' hd' c' hc'))
    refine (R.step n (v i) (v j) c).mpr (Or.inr ⟨v (i + (j - i) / 2), ?_, ?_⟩)
    · exact leg i (i + (j - i) / 2) hi (by omega) (by omega) (by omega)
    · exact leg (i + (j - i) / 2) j (by omega) (by omega) hj (by omega)

/-- A leg between inner items passes test c for c in G once it is at most 2^t links long. -/
theorem inner_leg (ok : CertOK net S C) (ch : Chain S pat v l) (R : Run net pat) (t i j : Nat)
    (hi : 1 ≤ i) (hij : i + 1 ≤ j) (hj : j + 1 ≤ l) (hd : j - i ≤ 2 ^ t) (c : Ch) (hc : C.G.get c = true) :
    net.test c (pat (v i) (v j)) (R.L t (v i) (v j)) = true := by
  by_cases hadj : j = i + 1
  · subst hadj
    exact ok.inner c hc _ (ch.inner i hi (by omega)) _
  · rw [ch.II i j hi (by omega) hj]
    exact ok.closeII c hc _ (Bits.le_of_get (fun c' hc' => inner_double ok ch R t i j hi (by omega) hj hd c' hc'))

/-- The right cap: pairs (inner item, Y) carry the bits of GY one loop later. -/
theorem cap_IY (ok : CertOK net S C) (ch : Chain S pat v l) (R : Run net pat) (t i : Nat)
    (hi : 1 ≤ i) (hil : i + 2 ≤ l) (hd : l - 1 - i ≤ 2 ^ t) (c : Ch) (hc : C.GY.get c = true) :
    (R.L (t + 1) (v i) (v l)).get c = true := by
  have hcG : C.G.get c = true := Bits.get_of_le ok.gy_le c hc
  refine (R.step t (v i) (v l) c).mpr (Or.inr ⟨v (l - 1), ?_, ?_⟩)
  · exact inner_leg ok ch R t i (l - 1) hi (by omega) (by omega) hd c hcG
  · exact ok.gy_last c hc _ ch.last _

/-- The left cap: pairs (X, inner item) carry the bits of GX one loop later. -/
theorem cap_XI (ok : CertOK net S C) (ch : Chain S pat v l) (R : Run net pat) (t j : Nat)
    (hj : 2 ≤ j) (hjl : j + 1 ≤ l) (hd : j - 1 ≤ 2 ^ t) (c : Ch) (hc : C.GX.get c = true) :
    (R.L (t + 1) (v 0) (v j)).get c = true := by
  have hcG : C.G.get c = true := Bits.get_of_le ok.gx_le c hc
  refine (R.step t (v 0) (v j) c).mpr (Or.inr ⟨v 1, ?_, ?_⟩)
  · exact ok.gx_first c hc _ ch.first _
  · exact inner_leg ok ch R t 1 j (by omega) (by omega) hjl hd c hcG

/-- Completeness on a chain. A table that passes `checkCert` answers yes for every chain of ℓ links with the patterns of
`S`, by loop t + 2 where 2^t ≥ ℓ - 2, and at every later loop. -/
theorem chain_complete (ok : CertOK net S C) (ch : Chain S pat v l) (R : Run net pat) (t : Nat) (hl : l ≤ 2 ^ t + 2) :
    ∀ t', t + 2 ≤ t' → net.read (R.L t' (v 0) (v l)) = true := by
  intro t' ht'
  have hlen := ch.len
  by_cases h2 : l = 2
  · -- two links: X - v 1 - Y
    apply ok.read_short
    apply Bits.le_of_get
    intro c hc
    have h1 : (R.L 1 (v 0) (v l)).get c = true := by
      refine (R.step 0 (v 0) (v l) c).mpr (Or.inr ⟨v 1, ?_, ?_⟩)
      · exact ok.rs_first c hc _ ch.first _
      · have := ch.last
        rw [h2] at this
        rw [h2]
        exact ok.rs_last c hc _ this _
    exact R.mono h1 t' (by omega)
  · apply ok.read_long
    apply Bits.le_of_get
    intro c hc
    rcases (Bits.union_get c).mp hc with hr | hlft
    · -- right cap: (X, v 1) is the first link, (v 1, Y) carries GY
      have h1 : (R.L (t + 2) (v 0) (v l)).get c = true := by
        refine (R.step (t + 1) (v 0) (v l) c).mpr (Or.inr ⟨v 1, ?_, ?_⟩)
        · exact ok.rr_first c hr _ ch.first _
        · rw [ch.IY 1 (by omega) (by omega)]
          exact ok.rr_IY c hr _ (Bits.le_of_get (fun c' hc' => cap_IY ok ch R t 1 (by omega) (by omega) (by omega) c' hc'))
      exact R.mono h1 t' ht'
    · -- left cap: (X, v (ℓ-1)) carries GX, (v (ℓ-1), Y) is the last link
      have h1 : (R.L (t + 2) (v 0) (v l)).get c = true := by
        refine (R.step (t + 1) (v 0) (v l) c).mpr (Or.inr ⟨v (l - 1), ?_, ?_⟩)
        · rw [ch.XI (l - 1) (by omega) (by omega)]
          exact ok.rl_XI c hlft _ (Bits.le_of_get (fun c' hc' => cap_XI ok ch R t (l - 1) (by omega) (by omega) (by omega) c' hc'))
        · exact ok.rl_last c hlft _ ch.last _
      exact R.mono h1 t' ht'

end chain

/-! ## Walks and shortcuts -/

/-- v 0, ..., v ℓ is a chain of links from a to b. -/
def IsWalk (link : α → α → Prop) (v : Nat → α) (l : Nat) (a b : α) : Prop :=
  v 0 = a ∧ v l = b ∧ ∀ i, i < l → link (v i) (v (i + 1))

/-- No two items of the chain that are two or more steps apart are linked directly. -/
def ShortcutFree (link : α → α → Prop) (v : Nat → α) (l : Nat) : Prop :=
  ∀ i j, i + 2 ≤ j → j ≤ l → ¬ link (v i) (v j)

theorem Reach.toWalk {link : α → α → Prop} {a b : α} (h : Reach link a b) : ∃ v l, IsWalk link v l a b := by
  induction h with
  | refl => exact ⟨fun _ => a, 0, rfl, rfl, fun i hi => absurd hi (by omega)⟩
  | @tail b c _ hl ih =>
    obtain ⟨v, l, h0, hlast, hstep⟩ := ih
    refine ⟨fun k => if k ≤ l then v k else c, l + 1, ?_, ?_, ?_⟩
    · simp [h0]
    · show (if l + 1 ≤ l then v (l + 1) else c) = c
      have hn : ¬ (l + 1 ≤ l) := by omega
      simp [hn]
    · intro i hi
      by_cases hil : i + 1 ≤ l
      · have h1 : i ≤ l := by omega
        simp only [h1, hil, ite_true]
        exact hstep i (by omega)
      · have h1 : i = l := by omega
        subst h1
        simp only [Nat.le_refl, ite_true]
        have : ¬ (i + 1 ≤ i) := by omega
        simp only [this, ite_false]
        rw [hlast]
        exact hl

theorem IsWalk.reach {link : α → α → Prop} {v : Nat → α} {l : Nat} {a b : α} (h : IsWalk link v l a b) :
    Reach link a b := by
  obtain ⟨h0, hl, hs⟩ := h
  have key : ∀ n, n ≤ l → Reach link (v 0) (v n) := by
    intro n
    induction n with
    | zero => intro _; exact Reach.refl _
    | succ m ih => intro hm; exact Reach.tail (ih (by omega)) (hs m (by omega))
  rw [← h0, ← hl]
  exact key l (Nat.le_refl l)

/-- A sub-chain of a walk is a chain of links. -/
theorem IsWalk.reach_between {link : α → α → Prop} {v : Nat → α} {l : Nat} {a b : α} (h : IsWalk link v l a b)
    (i j : Nat) (hij : i ≤ j) (hj : j ≤ l) : Reach link (v i) (v j) := by
  obtain ⟨_, _, hs⟩ := h
  have key : ∀ d, i + d ≤ l → Reach link (v i) (v (i + d)) := by
    intro d
    induction d with
    | zero => intro _; exact Reach.refl _
    | succ m ih => intro hm; exact Reach.tail (ih (by omega)) (hs (i + m) (by omega))
  have := key (j - i) (by omega)
  have e : i + (j - i) = j := by omega
  rw [e] at this
  exact this

/-- Every walk can be cut down to a walk without shortcuts that is no longer. -/
theorem exists_shortcut_free {link : α → α → Prop} {a b : α} :
    ∀ l (v : Nat → α), IsWalk link v l a b → ∃ v' l', l' ≤ l ∧ IsWalk link v' l' a b ∧ ShortcutFree link v' l' := by
  intro l
  induction l using Nat.strongRecOn with
  | _ l ih =>
    intro v hw
    by_cases hsc : ShortcutFree link v l
    · exact ⟨v, l, Nat.le_refl l, hw, hsc⟩
    · have : ∃ i j, i + 2 ≤ j ∧ j ≤ l ∧ link (v i) (v j) := by
        apply Classical.byContradiction
        intro hne
        apply hsc
        intro i j h1 h2 h3
        exact hne ⟨i, j, h1, h2, h3⟩
      obtain ⟨i, j, hij, hjl, hlink⟩ := this
      obtain ⟨h0, hlast, hstep⟩ := hw
      let v' : Nat → α := fun k => if k ≤ i then v k else v (k + (j - i - 1))
      have hw' : IsWalk link v' (l - (j - i - 1)) a b := by
        refine ⟨?_, ?_, ?_⟩
        · show (if 0 ≤ i then v 0 else v (0 + (j - i - 1))) = a
          simp [h0]
        · show (if l - (j - i - 1) ≤ i then v (l - (j - i - 1)) else v (l - (j - i - 1) + (j - i - 1))) = b
          have h1 : ¬ (l - (j - i - 1) ≤ i) := by omega
          have h2 : l - (j - i - 1) + (j - i - 1) = l := by omega
          simp only [h1, ite_false, h2]
          exact hlast
        · intro k hk
          show link (if k ≤ i then v k else v (k + (j - i - 1)))
            (if k + 1 ≤ i then v (k + 1) else v (k + 1 + (j - i - 1)))
          by_cases hki : k + 1 ≤ i
          · have h1 : k ≤ i := by omega
            simp only [h1, hki, ite_true]
            exact hstep k (by omega)
          · by_cases hke : k = i
            · subst hke
              have h2 : k + 1 + (j - k - 1) = j := by omega
              simp only [Nat.le_refl, ite_true, hki, ite_false, h2]
              exact hlink
            · have h1 : ¬ (k ≤ i) := by omega
              have h2 : k + 1 + (j - i - 1) = k + (j - i - 1) + 1 := by omega
              simp only [h1, hki, ite_false, h2]
              exact hstep (k + (j - i - 1)) (by omega)
      obtain ⟨v'', l'', hle, hw'', hsf⟩ := ih (l - (j - i - 1)) (by omega) v' hw'
      exact ⟨v'', l'', by omega, hw'', hsf⟩

end generic

/-! ## The kinship layout `within` (two direction classes) -/

variable {α : Type}

/-- An instance in the layout `within`: the items are the mentions of people; `x`, `y` are the question's two names.
`sp a b`: a and b name the same person (and are different mentions), as the read-in gives it. `stmt a b`: a and b are the
two names of one statement. `cls1 a`, `slot1 a`: the class of a's statement (false: its first-named person is read as the
parent) and whether a is named second. -/
structure Inst (α : Type) where
  x : α
  y : α
  sp : α → α → Bool
  stmt : α → α → Bool
  cls1 : α → Bool
  slot1 : α → Bool
  x_ne_y : x ≠ y
  sp_irrefl : ∀ a, sp a a = false
  stmt_irrefl : ∀ a, stmt a a = false
  /-- statements do not use the question's two items -/
  stmt_notq : ∀ a b, stmt a b = true → a ≠ x ∧ a ≠ y ∧ b ≠ x ∧ b ≠ y
  /-- a statement names two different people -/
  stmt_notsp : ∀ a b, stmt a b = true → sp a b = false
  /-- the question asks about two different people -/
  sp_xy : sp x y = false ∧ sp y x = false

namespace Inst
variable (I : Inst α)

/-- a is the mention its statement's reading calls the parent -/
def isParent (a : α) : Bool := I.cls1 a == I.slot1 a

/-- The pattern of a pair, exactly as `organ.Layout.pid` computes it for links `within`, K = 2. -/
noncomputable def pat (a b : α) : Nat :=
  if a = b then (if a = I.x ∨ a = I.y then 7 else 1)
  else
    (if I.stmt a b = true then 1 + 2 * (I.cls1 a).toNat + (I.slot1 a).toNat
     else if a = I.x ∧ b = I.y then 5 else if a = I.y ∧ b = I.x then 6 else 0) * 16
    + (if I.sp a b = true then 8 else 0) + (if a = I.x ∨ a = I.y then 4 else 0) + (if b = I.x ∨ b = I.y then 2 else 0)

/-- A link: the same person, or parent → child inside one statement. -/
def link (a b : α) : Prop := I.sp a b = true ∨ (I.stmt a b = true ∧ I.isParent a = true)

/-- Nobody is their own ancestor: the child of a statement is not linked back to that statement's parent. -/
def Acyclic : Prop := ∀ a b, I.stmt a b = true → I.isParent a = false → ¬ Reach I.link a b

end Inst

/-- The roles of patterns in the layout `within` with two classes. -/
def within : Shape := { first := [12], last := [10], inner := [8, 16, 64], pII := 0, pXI := 4, pIY := 2 }

/-- The 14 patterns that occur in a well-formed instance; the links among them; the pairs (a, a) among them. -/
def withinOcc : List Nat := [0, 1, 2, 4, 7, 8, 10, 12, 16, 32, 48, 64, 86, 102]
def withinLink : List Nat := [8, 10, 12, 16, 64]
def withinId : List Nat := [1, 7]

/-- The finite soundness check: with no kept bit, a test passes only on a link pattern or on a pattern of a pair (a, a);
with no kept bit the read-out says no. -/
def checkSound (net : Net) : Bool :=
  Ch.all (fun c => withinOcc.all fun p => !(net.test c p Bits.zero) || (withinLink.contains p || withinId.contains p)) &&
  !(net.read Bits.zero)

/-- The same finite soundness check for any lists of occurring / link / self patterns. Used for the layout `partner`,
whose pattern lemmas are not formalised in this file. -/
def checkSoundOn (occ link ids : List Nat) (net : Net) : Bool :=
  Ch.all (fun c => occ.all fun p => !(net.test c p Bits.zero) || (link.contains p || ids.contains p)) &&
  !(net.read Bits.zero)

/-- The roles of patterns in the layout `partner` with two classes: MEASURED on generated instances by
the certificate script (which patterns occur, which are links, which sit where on a shortest chain); not derived
in Lean. With these roles, `Run.sound` and `chain_complete` apply to a `partner` table as they do to a `within` table. -/
def partner : Shape := { first := [12], last := [18, 66], inner := [16, 64], pII := 0, pXI := 4, pIY := 2 }
def partnerOcc : List Nat := [0, 1, 2, 4, 7, 8, 10, 12, 16, 18, 32, 34, 48, 50, 64, 66, 84, 86, 100, 102]
def partnerLink : List Nat := [8, 10, 12, 16, 18, 64, 66]
def partnerId : List Nat := [1, 7]

section within
variable (I : Inst α)

/-! ### The value of the pattern, case by case -/

theorem pat_self (a : α) : I.pat a a = 7 ∨ I.pat a a = 1 := by
  unfold Inst.pat
  by_cases hq : a = I.x ∨ a = I.y
  · left; simp [hq]
  · right; simp [hq]

theorem pat_stmt {a b : α} (hst : I.stmt a b = true) :
    I.pat a b = 16 * (1 + 2 * (I.cls1 a).toNat + (I.slot1 a).toNat) := by
  obtain ⟨h1, h2, h3, h4⟩ := I.stmt_notq a b hst
  have hsp := I.stmt_notsp a b hst
  have hab : a ≠ b := by
    intro h
    subst h
    have := I.stmt_irrefl a
    rw [this] at hst
    exact Bool.noConfusion hst
  unfold Inst.pat
  simp [hab, hst, h1, h2, h3, h4, hsp]
  omega

theorem pat_xy : I.pat I.x I.y = 86 := by
  have h1 : I.stmt I.x I.y = false := by
    cases h : I.stmt I.x I.y
    · rfl
    · exact absurd rfl (I.stmt_notq _ _ h).1
  unfold Inst.pat
  simp [I.x_ne_y, h1, I.sp_xy.1]

theorem pat_yx : I.pat I.y I.x = 102 := by
  have hne : I.y ≠ I.x := fun h => I.x_ne_y h.symm
  have h1 : I.stmt I.y I.x = false := by
    cases h : I.stmt I.y I.x
    · rfl
    · exact absurd rfl (I.stmt_notq _ _ h).2.1
  unfold Inst.pat
  simp [hne, h1, I.sp_xy.2]

theorem pat_generic {a b : α} (hab : a ≠ b) (hst : I.stmt a b = false) (hxy : ¬ (a = I.x ∧ b = I.y))
    (hyx : ¬ (a = I.y ∧ b = I.x)) :
    I.pat a b = (if I.sp a b = true then 8 else 0) + (if a = I.x ∨ a = I.y then 4 else 0)
      + (if b = I.x ∨ b = I.y then 2 else 0) := by
  unfold Inst.pat
  simp [hab, hst, hxy, hyx]

/-- Two different items that are not both in the question, with no statement between them. -/
theorem not_both {a b : α} (hab : a ≠ b) (hxy : ¬ (a = I.x ∧ b = I.y)) (hyx : ¬ (a = I.y ∧ b = I.x)) :
    ¬ ((a = I.x ∨ a = I.y) ∧ (b = I.x ∨ b = I.y)) := by
  rintro ⟨ha | ha, hb | hb⟩
  · exact hab (ha.trans hb.symm)
  · exact hxy ⟨ha, hb⟩
  · exact hyx ⟨ha, hb⟩
  · exact hab (ha.trans hb.symm)

/-- The six values of the pattern of two different items with no statement between them, not both in the question. -/
theorem generic_six (sp : Bool) (p q : Prop) [Decidable p] [Decidable q] (hnb : ¬ (p ∧ q)) :
    ((if sp = true then 8 else 0) + (if p then 4 else 0) + (if q then 2 else 0) = 0 ∧ sp = false ∧ ¬ p ∧ ¬ q) ∨
    ((if sp = true then 8 else 0) + (if p then 4 else 0) + (if q then 2 else 0) = 2 ∧ sp = false ∧ ¬ p ∧ q) ∨
    ((if sp = true then 8 else 0) + (if p then 4 else 0) + (if q then 2 else 0) = 4 ∧ sp = false ∧ p ∧ ¬ q) ∨
    ((if sp = true then 8 else 0) + (if p then 4 else 0) + (if q then 2 else 0) = 8 ∧ sp = true ∧ ¬ p ∧ ¬ q) ∨
    ((if sp = true then 8 else 0) + (if p then 4 else 0) + (if q then 2 else 0) = 10 ∧ sp = true ∧ ¬ p ∧ q) ∨
    ((if sp = true then 8 else 0) + (if p then 4 else 0) + (if q then 2 else 0) = 12 ∧ sp = true ∧ p ∧ ¬ q) := by
  by_cases hp : p
  · by_cases hq : q
    · exact absurd ⟨hp, hq⟩ hnb
    · rw [if_pos hp, if_neg hq]
      cases sp
      · exact Or.inr (Or.inr (Or.inl ⟨by decide, rfl, hp, hq⟩))
      · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨by decide, rfl, hp, hq⟩))))
  · by_cases hq : q
    · rw [if_neg hp, if_pos hq]
      cases sp
      · exact Or.inr (Or.inl ⟨by decide, rfl, hp, hq⟩)
      · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨by decide, rfl, hp, hq⟩))))
    · rw [if_neg hp, if_neg hq]
      cases sp
      · exact Or.inl ⟨by decide, rfl, hp, hq⟩
      · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨by decide, rfl, hp, hq⟩)))

/-- Every pair's pattern is one of these kinds. -/
theorem pat_kinds (a b : α) :
    (a = b ∧ (I.pat a b = 7 ∨ I.pat a b = 1)) ∨
    (I.stmt a b = true ∧ I.pat a b = 16 * (1 + 2 * (I.cls1 a).toNat + (I.slot1 a).toNat)) ∨
    I.pat a b = 86 ∨ I.pat a b = 102 ∨
    (I.pat a b = 0 ∨ I.pat a b = 2 ∨ I.pat a b = 4) ∨
    ((I.pat a b = 8 ∨ I.pat a b = 10 ∨ I.pat a b = 12) ∧ I.sp a b = true) := by
  by_cases hab : a = b
  · subst hab
    exact Or.inl ⟨rfl, pat_self I a⟩
  · cases hst : I.stmt a b
    · by_cases hxy : a = I.x ∧ b = I.y
      · obtain ⟨rfl, rfl⟩ := hxy
        exact Or.inr (Or.inr (Or.inl (pat_xy I)))
      · by_cases hyx : a = I.y ∧ b = I.x
        · obtain ⟨rfl, rfl⟩ := hyx
          exact Or.inr (Or.inr (Or.inr (Or.inl (pat_yx I))))
        · have e := pat_generic I hab hst hxy hyx
          rcases generic_six (I.sp a b) (a = I.x ∨ a = I.y) (b = I.x ∨ b = I.y) (not_both I hab hxy hyx) with
            ⟨h, _⟩ | ⟨h, _⟩ | ⟨h, _⟩ | ⟨h, hs, _⟩ | ⟨h, hs, _⟩ | ⟨h, hs, _⟩
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Or.inl (e.trans h))))))
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Or.inr (Or.inl (e.trans h)))))))
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Or.inr (Or.inr (e.trans h)))))))
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨Or.inl (e.trans h), hs⟩))))
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨Or.inr (Or.inl (e.trans h)), hs⟩))))
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨Or.inr (Or.inr (e.trans h)), hs⟩))))
    · exact Or.inr (Or.inl ⟨rfl, pat_stmt I hst⟩)

theorem pat_occ (a b : α) : I.pat a b ∈ withinOcc := by
  rcases pat_kinds I a b with ⟨_, h | h⟩ | ⟨_, h⟩ | h | h | (h | h | h) | ⟨h | h | h, _⟩
  all_goals first
    | (rw [h]; decide)
    | (rw [h]; cases I.cls1 a <;> cases I.slot1 a <;> decide)

theorem pat_link {a b : α} (h : I.pat a b ∈ withinLink) : I.link a b := by
  rcases pat_kinds I a b with ⟨_, e | e⟩ | ⟨hst, e⟩ | e | e | (e | e | e) | ⟨_, hsp⟩
  · rw [e] at h; exact absurd h (by decide)
  · rw [e] at h; exact absurd h (by decide)
  · rw [e] at h
    cases hc : I.cls1 a <;> cases hs : I.slot1 a <;> rw [hc, hs] at h
    · exact Or.inr ⟨hst, by simp [Inst.isParent, hc, hs]⟩
    · exact absurd h (by decide)
    · exact absurd h (by decide)
    · exact Or.inr ⟨hst, by simp [Inst.isParent, hc, hs]⟩
  · rw [e] at h; exact absurd h (by decide)
  · rw [e] at h; exact absurd h (by decide)
  · rw [e] at h; exact absurd h (by decide)
  · rw [e] at h; exact absurd h (by decide)
  · rw [e] at h; exact absurd h (by decide)
  · exact Or.inl hsp

theorem pat_id {a b : α} (h : I.pat a b ∈ withinId) : a = b := by
  rcases pat_kinds I a b with ⟨hab, _⟩ | ⟨_, e⟩ | e | e | (e | e | e) | ⟨e | e | e, _⟩
  · exact hab
  · rw [e] at h
    cases hc : I.cls1 a <;> cases hs : I.slot1 a <;> rw [hc, hs] at h <;> exact absurd h (by decide)
  all_goals (rw [e] at h; exact absurd h (by decide))

/-- Soundness for the layout: a table that passes `checkSound` never says yes unless X is linked to Y. -/
theorem within_sound {net : Net} (hS : checkSound net = true) (R : Run net I.pat) (t : Nat)
    (h : net.read (R.L t I.x I.y) = true) : Reach I.link I.x I.y := by
  simp only [checkSound, Bool.and_eq_true] at hS
  obtain ⟨hT, hR⟩ := hS
  have h0 : ∀ c a b, net.test c (I.pat a b) Bits.zero = true → I.link a b ∨ a = b := by
    intro c a b hp
    have := imp_of_or (List.all_eq_true.mp (Ch.all_spec hT c) (I.pat a b) (pat_occ I a b)) hp
    simp only [Bool.or_eq_true, List.contains_iff_mem] at this
    rcases this with hl | hi
    · exact Or.inl (pat_link I hl)
    · exact Or.inr (pat_id I hi)
  have hz : R.L t I.x I.y ≠ Bits.zero := by
    intro hz
    rw [hz] at h
    simp [h] at hR
  obtain ⟨c, hc⟩ := Bits.exists_get_of_ne_zero hz
  exact R.sound h0 t I.x I.y c hc

/-! ### The same soundness, through the two-item dual (r = reachable from X, n = the rest) -/

/-- The dual of "is Y linked from X" for the layout `within`: two items, `true` (reachable from X) and `false`. A pair
(reachable, unreachable) can carry no kept bit and no link pattern; every other pair is unconstrained. -/
def withinDual : Dual Bool where
  B := fun p q => if p = true ∧ q = false then Bits.zero else ⟨true, true, true⟩
  allowed := fun p q π => π ∈ withinOcc ∧ (p = true → q = false → π ∉ withinLink ∧ π ∉ withinId)

/-- `checkSound` is the simulation table of the two-item dual. -/
theorem withinDual_closed {net : Net} (hS : checkSound net = true) : withinDual.Closed net := by
  simp only [checkSound, Bool.and_eq_true] at hS
  obtain ⟨hT, _⟩ := hS
  have key : ∀ (c : Ch) (π : Nat), π ∈ withinOcc → π ∉ withinLink → π ∉ withinId →
      net.test c π Bits.zero = true → False := by
    intro c π hocc hl hi hp
    have := imp_of_or (List.all_eq_true.mp (Ch.all_spec hT c) π hocc) hp
    simp only [Bool.or_eq_true, List.contains_iff_mem] at this
    exact this.elim hl hi
  intro c p m q π1 π2 s1 s2 h1 h2 hs1 hs2 ht1 ht2
  show (if p = true ∧ q = false then Bits.zero else (⟨true, true, true⟩ : Bits)).get c = true
  by_cases hpq : p = true ∧ q = false
  · exfalso
    obtain ⟨hp, hq⟩ := hpq
    cases m
    · -- the middle item is unreachable: the first leg goes from reachable to unreachable
      have hz : s1 = Bits.zero := by
        apply Bits.eq_zero_of_le_zero
        have e : withinDual.B p false = Bits.zero := by simp [withinDual, hp]
        rw [← e]; exact hs1
      rw [hz] at ht1
      exact key c π1 h1.1 (h1.2 hp rfl).1 (h1.2 hp rfl).2 ht1
    · -- the middle item is reachable: the second leg goes from reachable to unreachable
      have hz : s2 = Bits.zero := by
        apply Bits.eq_zero_of_le_zero
        have e : withinDual.B true q = Bits.zero := by simp [withinDual, hq]
        rw [← e]; exact hs2
      rw [hz] at ht2
      exact key c π2 h2.1 (h2.2 rfl hq).1 (h2.2 rfl hq).2 ht2
  · rw [if_neg hpq]
    cases c <;> rfl

/-- No false yes, by the dual: if Y is not linked from X, the read-out says no at every loop, on an instance of any
size. (The contrapositive of `within_sound`, proved through `Run.no_false_yes`.) -/
theorem within_no_false_yes {net : Net} (hS : checkSound net = true) (R : Run net I.pat)
    (hno : ¬ Reach I.link I.x I.y) (t : Nat) : net.read (R.L t I.x I.y) = false := by
  have hR : net.read Bits.zero = false := by
    simp only [checkSound, Bool.and_eq_true] at hS
    have := hS.2
    cases hr : net.read Bits.zero
    · rfl
    · rw [hr] at this
      exact absurd this (by decide)
  let h : α → Bool := fun a => decide (Reach I.link I.x a)
  have hh : ∀ a, h a = true ↔ Reach I.link I.x a := fun a => by simp [h]
  have hpat : ∀ a b, withinDual.allowed (h a) (h b) (I.pat a b) := by
    intro a b
    refine ⟨pat_occ I a b, ?_⟩
    intro ha hb
    have ra : Reach I.link I.x a := (hh a).mp ha
    have nb : ¬ Reach I.link I.x b := by
      intro rb
      have := (hh b).mpr rb
      rw [hb] at this
      exact Bool.noConfusion this
    refine ⟨fun hl => nb (Reach.tail ra (pat_link I hl)), fun hi => ?_⟩
    have e := pat_id I hi
    rw [e] at ra
    exact nb ra
  apply R.no_false_yes withinDual (withinDual_closed hS) h hpat I.x I.y
  intro s hs
  have hx : h I.x = true := (hh I.x).mpr (Reach.refl I.x)
  have hy : h I.y = false := by
    cases hy : h I.y
    · rfl
    · exact absurd ((hh I.y).mp hy) hno
  have e : withinDual.B (h I.x) (h I.y) = Bits.zero := by simp [withinDual, hx, hy]
  rw [e] at hs
  rw [Bits.eq_zero_of_le_zero hs]
  exact hR

/-! ### Completeness: a shortest chain of links carries the patterns of `within` -/

theorem stmt_false_of_q {a b : α} (h : a = I.x ∨ a = I.y ∨ b = I.x ∨ b = I.y) : I.stmt a b = false := by
  cases hst : I.stmt a b
  · rfl
  · obtain ⟨h1, h2, h3, h4⟩ := I.stmt_notq a b hst
    rcases h with h | h | h | h
    · exact absurd h h1
    · exact absurd h h2
    · exact absurd h h3
    · exact absurd h h4

/-- A link out of X, or into Y, is a "same person" link. -/
theorem link_from_x {b : α} (h : I.link I.x b) : I.sp I.x b = true := by
  rcases h with h | ⟨h, _⟩
  · exact h
  · rw [stmt_false_of_q I (Or.inl rfl)] at h
    exact Bool.noConfusion h

theorem link_to_y {a : α} (h : I.link a I.y) : I.sp a I.y = true := by
  rcases h with h | ⟨h, _⟩
  · exact h
  · rw [stmt_false_of_q I (Or.inr (Or.inr (Or.inr rfl)))] at h
    exact Bool.noConfusion h

theorem bool_false_of_not {b : Bool} (h : ¬ b = true) : b = false := by
  cases b
  · rfl
  · exact absurd rfl h

theorem within_chain (hac : I.Acyclic) {v : Nat → α} {l : Nat} (hw : IsWalk I.link v l I.x I.y)
    (hsf : ShortcutFree I.link v l) : Chain within I.pat v l := by
  obtain ⟨h0, hl, hstep⟩ := hw
  have hw' : IsWalk I.link v l I.x I.y := ⟨h0, hl, hstep⟩
  -- inner items are neither X nor Y
  have notx : ∀ i, 1 ≤ i → i ≤ l → v i ≠ I.x := by
    intro i hi hil hx
    by_cases hlt : i < l
    · have := hstep i hlt
      rw [hx, ← h0] at this
      exact hsf 0 (i + 1) (by omega) (by omega) this
    · have : i = l := by omega
      rw [this, hl] at hx
      exact I.x_ne_y hx.symm
  have noty : ∀ i, i + 1 ≤ l → v i ≠ I.y := by
    intro i hil hy
    by_cases hpos : 1 ≤ i
    · have := hstep (i - 1) (by omega)
      have e : i - 1 + 1 = i := by omega
      rw [e, hy, ← hl] at this
      exact hsf (i - 1) l (by omega) (Nat.le_refl l) this
    · have : i = 0 := by omega
      rw [this, h0] at hy
      exact I.x_ne_y hy
  have hlen : 2 ≤ l := by
    apply Classical.byContradiction
    intro hlt
    have hcases : l = 0 ∨ l = 1 := by omega
    rcases hcases with e | e
    · rw [e] at hl
      rw [h0] at hl
      exact I.x_ne_y hl
    · have := hstep 0 (by omega)
      rw [e] at hl
      rw [h0] at this
      simp only [Nat.zero_add] at this
      rw [hl] at this
      have := link_from_x I this
      rw [I.sp_xy.1] at this
      exact Bool.noConfusion this
  -- the value of the pattern on a generic pair
  have gen : ∀ a b, a ≠ b → I.stmt a b = false → ¬ (a = I.x ∧ b = I.y) → ¬ (a = I.y ∧ b = I.x) →
      ∀ n, ((if I.sp a b = true then 8 else 0) + (if a = I.x ∨ a = I.y then 4 else 0)
        + (if b = I.x ∨ b = I.y then 2 else 0) = n) → I.pat a b = n := by
    intro a b hab hst hxy hyx n hn
    rw [pat_generic I hab hst hxy hyx]
    exact hn
  refine ⟨hlen, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- first link
    have hlk := hstep 0 (by omega)
    simp only [Nat.zero_add] at hlk
    rw [h0] at hlk
    have hsp := link_from_x I hlk
    have h1x := notx 1 (by omega) (by omega)
    have h1y := noty 1 (by omega)
    have e : I.pat (v 0) (v 1) = 12 := by
      rw [h0]
      apply gen I.x (v 1) (fun h => h1x h.symm) (stmt_false_of_q I (Or.inl rfl)) (fun h => h1y h.2)
        (fun h => I.x_ne_y h.1)
      have hb : ¬ (v 1 = I.x ∨ v 1 = I.y) := fun h => h.elim h1x h1y
      rw [if_pos hsp, if_pos (Or.inl rfl), if_neg hb]
    rw [e]
    decide
  · -- last link
    have hlk := hstep (l - 1) (by omega)
    have e1 : l - 1 + 1 = l := by omega
    rw [e1, hl] at hlk
    have hsp := link_to_y I hlk
    have hax := notx (l - 1) (by omega) (by omega)
    have hay := noty (l - 1) (by omega)
    have e : I.pat (v (l - 1)) (v l) = 10 := by
      rw [hl]
      apply gen (v (l - 1)) I.y hay (stmt_false_of_q I (Or.inr (Or.inr (Or.inr rfl)))) (fun h => hax h.1)
        (fun h => hay h.1)
      have ha : ¬ (v (l - 1) = I.x ∨ v (l - 1) = I.y) := fun h => h.elim hax hay
      rw [if_pos hsp, if_neg ha, if_pos (Or.inr rfl)]
    rw [e]
    decide
  · -- inner links
    intro i hi hil
    have hax := notx i hi (by omega)
    have hay := noty i (by omega)
    have hbx := notx (i + 1) (by omega) (by omega)
    have hby := noty (i + 1) (by omega)
    have ha : ¬ (v i = I.x ∨ v i = I.y) := fun h => h.elim hax hay
    have hb : ¬ (v (i + 1) = I.x ∨ v (i + 1) = I.y) := fun h => h.elim hbx hby
    rcases hstep i (by omega) with hsp | ⟨hst, hpar⟩
    · have hab : v i ≠ v (i + 1) := by
        intro h
        rw [← h, I.sp_irrefl] at hsp
        exact Bool.noConfusion hsp
      have hst : I.stmt (v i) (v (i + 1)) = false := by
        cases hst : I.stmt (v i) (v (i + 1))
        · rfl
        · have := I.stmt_notsp _ _ hst
          rw [hsp] at this
          exact Bool.noConfusion this
      have e : I.pat (v i) (v (i + 1)) = 8 := by
        apply gen _ _ hab hst (fun h => hax h.1) (fun h => hay h.1)
        rw [if_pos hsp, if_neg ha, if_neg hb]
      rw [e]
      decide
    · rw [pat_stmt I hst]
      simp only [Inst.isParent] at hpar
      cases hc : I.cls1 (v i) <;> cases hs : I.slot1 (v i) <;> rw [hc, hs] at hpar
      · decide
      · exact absurd hpar (by decide)
      · exact absurd hpar (by decide)
      · decide
  · -- two inner items that are not neighbours: no relation at all
    intro i j hi hij hj
    have hax := notx i hi (by omega)
    have hay := noty i (by omega)
    have hbx := notx j (by omega) (by omega)
    have hby := noty j (by omega)
    have ha : ¬ (v i = I.x ∨ v i = I.y) := fun h => h.elim hax hay
    have hb : ¬ (v j = I.x ∨ v j = I.y) := fun h => h.elim hbx hby
    have hab : v i ≠ v j := by
      intro h
      have := hstep j (by omega)
      rw [← h] at this
      exact hsf i (j + 1) (by omega) (by omega) this
    have hsp : I.sp (v i) (v j) = false :=
      bool_false_of_not (fun h => hsf i j hij (by omega) (Or.inl h))
    have hst : I.stmt (v i) (v j) = false := by
      cases hst : I.stmt (v i) (v j)
      · rfl
      · exfalso
        cases hpar : I.isParent (v i)
        · exact hac _ _ hst hpar (hw'.reach_between i j (by omega) (by omega))
        · exact hsf i j hij (by omega) (Or.inr ⟨hst, hpar⟩)
    show I.pat (v i) (v j) = 0
    apply gen _ _ hab hst (fun h => hax h.1) (fun h => hay h.1)
    rw [hsp, if_neg ha, if_neg hb]
    decide
  · -- (inner item, Y)
    intro i hi hil
    have hax := notx i hi (by omega)
    have hay := noty i (by omega)
    have ha : ¬ (v i = I.x ∨ v i = I.y) := fun h => h.elim hax hay
    have hsp : I.sp (v i) I.y = false := by
      apply bool_false_of_not
      intro h
      have : I.link (v i) (v l) := by rw [hl]; exact Or.inl h
      exact hsf i l hil (Nat.le_refl l) this
    show I.pat (v i) (v l) = 2
    rw [hl]
    apply gen _ _ hay (stmt_false_of_q I (Or.inr (Or.inr (Or.inr rfl)))) (fun h => hax h.1) (fun h => hay h.1)
    rw [hsp, if_neg ha, if_pos (Or.inr rfl)]
    decide
  · -- (X, inner item)
    intro j hj hjl
    have hbx := notx j (by omega) (by omega)
    have hby := noty j (by omega)
    have hb : ¬ (v j = I.x ∨ v j = I.y) := fun h => h.elim hbx hby
    have hsp : I.sp I.x (v j) = false := by
      apply bool_false_of_not
      intro h
      have : I.link (v 0) (v j) := by rw [h0]; exact Or.inl h
      exact hsf 0 j (by omega) (by omega) this
    show I.pat (v 0) (v j) = 4
    rw [h0]
    apply gen _ _ (fun h => hbx h.symm) (stmt_false_of_q I (Or.inl rfl)) (fun h => hby h.2) (fun h => I.x_ne_y h.1)
    rw [hsp, if_pos (Or.inl rfl), if_neg hb]
    decide

/-- **The trained block is right at every size.** For a table that passes the two finite checks, on every well-formed
instance of the layout `within`, with any number of items:
(1) the read-out never says yes unless X is linked to Y by a chain of "same person" and "parent → child" links;
(2) if nobody is their own ancestor and such a chain of ℓ links exists, the read-out says yes at loop t + 2 as soon as
    2^t ≥ ℓ - 2, and at every later loop. -/
theorem within_correct {net : Net} {C : Cert} (hS : checkSound net = true) (hC : checkCert net within C = true)
    (R : Run net I.pat) :
    (∀ t, net.read (R.L t I.x I.y) = true → Reach I.link I.x I.y) ∧
    (I.Acyclic → ∀ (v : Nat → α) (l : Nat), IsWalk I.link v l I.x I.y → ∀ t, l ≤ 2 ^ t + 2 →
      ∀ t', t + 2 ≤ t' → net.read (R.L t' I.x I.y) = true) := by
  refine ⟨fun t h => within_sound I hS R t h, ?_⟩
  intro hac v l hw t hlt t' ht'
  obtain ⟨v', l', hle, hw', hsf⟩ := exists_shortcut_free l v hw
  have ch := within_chain I hac hw' hsf
  have := chain_complete (checkCert_ok hC) ch R t (by omega) t' ht'
  rw [hw'.1, hw'.2.1] at this
  exact this

end within

end Organ
