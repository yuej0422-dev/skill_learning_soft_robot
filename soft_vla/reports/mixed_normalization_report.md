# Mixed Normalization Report

- Non-identity dimensions keep dataset mean/std normalization.
- Configured identity dimensions use processor mean=0/std=1.
- Identity indices: `{'observation.state': [12], 'action': [6]}`
- First processed state shape: `[1, 1, 25]`
- First processed action shape: `[1, 50, 19]`

## observation.state

- Original mean at identity indices: `[0.47448731406334005]`
- Original std at identity indices: `[0.4993486786367833]`
- Processor mean at identity indices: `[0.0]`
- Processor std at identity indices: `[1.0]`

## action

- Original mean at identity indices: `[0.47448731406334005]`
- Original std at identity indices: `[0.4993486786367833]`
- Processor mean at identity indices: `[0.0]`
- Processor std at identity indices: `[1.0]`
