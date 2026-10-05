# FINAL low-kappa_L DFT list: 15 crystals + 4 controls + 8 reserves

This is the final list; it replaces all earlier DFT lists. Checked 2026-10-06 (final
pre-DFT check, no problems found). Target: lattice thermal
conductivity kappa_L <= 1 W/m/K at 300 K. All numbers per crystal are in `index.csv`;
the 4 high-kappa controls are in `controls/` (predicted 25-103 W/m/K - DFT should put
them far above the 15, or the method is not separating low from high).

| #  | Formula      | ID         | Atoms | Space group  | Supercell (cell as given) | Spin-polarise     |
|----|--------------|------------|-------|--------------|---------------------------|-------------------|
| 1  | Cs4NbCrI12   | d7e7d897af | 18    | P4/mmm (123) | 2x2x1 (72)                | yes, Cr(III)      |
| 2  | Rb4HfPtI12   | e8a9252cd6 | 18    | P4/m (83)    | 2x2x1 (72)                | no                |
| 3  | Rb4ZrPtI12   | b7d6f18a56 | 18    | P4/m (83)    | 2x2x1 (72)                | no                |
| 4  | Rb4Ru2Cl10O  | 043233727a | 17    | I4/mmm (139) | 2x2x1 (68)                | check, Ru(IV)     |
| 5  | K4HfTeI12    | b80c9ce032 | 18    | P-1 (2)      | 2x2x1 (72)                | no                |
| 6  | Os(IF3)2     | a1da4cd64a | 18    | P-1 (2)      | 2x2x2 (144)               | yes, Os(V) + I2+  |
| 7  | K4HfMnI12    | 4130d5c336 | 18    | P-1 (2)      | 2x2x1 (72)                | yes, Mn(IV)       |
| 8  | TaAs(IF3)4   | 3c3cb17cb8 | 18    | Cm (8)       | 2x2x2 (144)               | check, I2+        |
| 9  | K4HfSnI12    | b40662d9a8 | 18    | P-1 (2)      | 2x2x1 (72)                | no                |
| 10 | K4ZrSnI12    | e43c4feb4e | 18    | P-1 (2)      | 2x2x1 (72)                | no                |
| 11 | K3Rb(TeI6)2  | 5deeb9c156 | 18    | P1 (1)       | 2x2x1 (72)                | no                |
| 12 | Rb5Au(BrO)2  | 33513b1219 | 20    | Pbam (55)    | 2x2x1 (80)                | no                |
| 13 | Rb2Ru(OF2)2  | be0bfba1fd | 18    | Ama2 (40)    | 2x2x2 (144)               | yes, Ru(VI)       |
| 14 | Cs4ZnNiO4    | dc325ace66 | 10    | C2/m (12)    | 3x2x2 (120)               | yes, Ni(II)       |
| 15 | K4Os2Cl10O   | 635c6436c7 | 17    | I4/mmm (139) | 2x2x1 (68)                | check, Os(IV)     |

Space groups: spglib, symprec 0.01. Supercells: smallest diagonal multiples of the cell AS
GIVEN with every lattice vector >= 10 A - re-pick after standardising a slanted cell.

## Model predictions in `index.csv`

kappa values are kappa_L in W/m/K at 300 K. `<model>` = ALIGNN, CGCNN-ens, newbase, which are
three independent predictors of the bulk (K) and shear (G) modulus.

| Column | What it is |
|---|---|
| `K_GPa_<model>`, `G_GPa_<model>` | the model's predicted moduli, in GPa |
| `kappa_mlip_<model>` | Slack model from those moduli + the MLIP phonon gamma (`gamma_mlip`) |
| `kappa_max3` | highest of the three `kappa_mlip_*`, used for the ranking |
| `kappa_typical`, `kappa_stress` | `kappa_max3` with gamma cut by 10% / 32%, floored at `kappa_cahill` |
| `gamma_poisson_<model>`, `kappa_poisson_<model>` | gamma from the model's own Poisson ratio, and the Slack kappa with it |
| `kappa_poisson_max3` | highest of the three `kappa_poisson_*` |
| `kappa_direct` | direct ALIGNN, structure -> kappa_L, trained on 6,641 PhoNIX DFT values |

`controls/index.csv` and `reserves/index.csv` have the same model columns.
`kappa_pred_slack_300K` in the controls file is the highest `kappa_mlip_*`.

## DFT setup notes

- **Slanted cells:** #5, #7, #9, #10, #11 have one angle near 55 deg; #13 has gamma = 135.8 deg.
  Standardise (Niggli / spglib standard cell) before relaxing.
- **Spin:** the known K salt of [Ru2OCl10]4- is diamagnetic (Ru-O-Ru pi bonding), so #4 and
  #15 probably relax to zero moment. #8's I2+ units have an odd electron count - check.
  DFT+U on the 3d ions (#1 Cr, #7 Mn, #14 Ni) if that is the group standard.
- **SOC:** I, Te, Pt, Os, Au, Hf, Ta are heavy - include or not per usual practice.
- **Symmetry quirks:** some spglib builds find NO symmetry for #13; a clean install gives Ama2
  at symprec 1e-3 to 0.1. #14 is P-1 at 1e-3 and C2/m at 1e-2 (slight distortion).
- **Imaginary MLIP modes to confirm with DFT phonons:** #5 -0.077 THz, #13 -0.062,
  #8 -0.060, #7 -0.018; the other 11 are within 0.001 THz of zero.
- **Chemistry:** #12 needs Au(+1); #6 and #8 are iodine-cation salts ([I2]+, I-I 2.63 A) of
  MF6- anions.
- **If time is short:** #2/#3 and #9/#10 are Hf/Zr twins - run one of each pair first.

## Reserves (CIFs in `reserves/`, numbered in order of use)

All 8 are tier 1, like the 15. Use first: res_01 Cs2CuAgO2, res_02 Fe3SbCl7O. The
direct ALIGNN model calls res_03-05 (LiHg2BrO2, Cs3(AgO2)2, LiHg2ClO2) high
(2.2-2.6 W/m/K). res_06-08 (Cs3MnO4F, Cs2KFeO4, Rb5Co(AuO)2) need rare oxidation states.

## How the 15 were predicted low

1. Slack model, MLIP Grueneisen gamma: all <= 0.85 W/m/K even with gamma cut to its worst
   plausible overestimate (`kappa_stress`).
2. Slack model, gamma from Poisson's ratio (step 92): 14/15 pass its stricter 0.54 cut;
   #14 misses at 0.57.
3. Direct ALIGNN trained on 6,641 PhoNIX DFT kappa_L values (transport_program crosswork 05):
   all <= 0.50; typical error about x1.6 on crystals like its training set.

None of the models has seen these crystals: treat the numbers as a ranking of where to
spend DFT, not as values DFT will reproduce.
