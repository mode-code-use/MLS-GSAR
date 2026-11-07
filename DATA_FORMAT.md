# Data format and split policy

The released model currently accepts **ten-person scenes only**. Native 12- or
15-person archives must not be silently reshaped or relabeled to fit it.

## Labeled scene archives

Training requires separate `train.npz` and `val.npz`; testing requires a third,
disjoint `test.npz`. Each is an ordinary non-pickled NPZ containing:

| Key | Shape and type | Meaning |
| --- | --- | --- |
| `signals` | `[B,10,200,6]`, finite numeric | Raw six-channel sensor windows; normalization is fitted on training scenes only. |
| `groups` | `[B,10]`, integer | Group identity within a scene. Use contiguous IDs starting at zero, 2-6 groups per scene. |
| `activities` | `[B,10]`, integer | Group behavior of each person, 0-9; -1 denotes noise. Every member of a group must have the same target activity. |
| `actions` | `[B,10]`, integer | Individual-action targets, 0-9, for the auxiliary action head. These are not inferred from group behavior. |
| `source_ids` | `[B,10]`, fixed-width string | Stable original window IDs, e.g. `collection:recording:window:person`; never assign a new ID just because a window is reused in another constructed scene. |

IDs and target labels are for auditing/supervision, never encoder inputs. Store
real provenance, not random IDs that conceal shared sources. Match channel order,
units and label mappings between all splits. The loader refuses nonfinite data,
fractional labels, inconsistent group activity labels, and unsupported shapes.

Split by participant/session/recording **before** constructing overlapping windows
where the source metadata permits it. The tools check repeated source IDs and
exact full/aligned nonconstant half-windows across partitions. These checks do
not establish participant/session independence when the original metadata is
missing, and do not detect every approximate or shifted duplicate.

## Adapter for `gar_data_ut.npz`

The provided adapter only accepts the previously audited five-person template
schema: `reshaped_segments` with shape `[rows,5,200,17]`, `segments_labels1` as
one template activity per row, and `segments_labels2` with shape `[rows,5]`.
It selects signal columns 0-5 and excludes columns 6-16. Physical channel names,
units and placement remain unverified; check the collection documentation.

```bash
python prepare_data.py --data /path/to/gar_data_ut.npz --output data/ut --accept-synthetic-scenes
```

The adapter uses the existing archive-order blocked per-class 62%/18%/20% split,
boundary purging, and exact full/half-window overlap removal. It then constructs
ten-person scenes **within each partition** from 3-4 distinct activity templates
plus a noise person. Group membership is assigned during construction, not
inferred by equality of individual action labels. All scene provenance and
source rows are recorded in `data_manifest.json` and `*_scene_provenance.npz`.
The ten-person group annotations are synthetic, not native collected ten-person
subgroup annotations. The default uses all retained labeled training rows;
`--labeled-ratio` can reduce the template pool used to construct training scenes.
In this public workflow SSL also uses those constructed training scenes; it does
not automatically add the remaining unselected windows as an unlabeled pool.

Generated files include `train.npz`, `val.npz`, `test.npz`, and the manifest.
The command refuses an existing output directory. Do not merge archives sharing
the same source signals into independent train/test sets. WBSensor and
`UTGroup_10.npz` are not supported by this adapter merely because of their names.

No input datasets are committed to the repository. A user with native compatible
scene annotations can provide the explicit schema above without using this
synthetic UT adapter.
