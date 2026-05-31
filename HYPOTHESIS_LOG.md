# Hypothesis Log

| Hypothesis | Result | Kept |
|---|---:|---|
| Large amount jumps versus a card's own history indicate compromise. | Strong signal in synthetic cases and top challenge alerts. | Yes |
| New merchant categories after a short card history are useful but noisy. | Useful when paired with amount, device, or country shifts. | Yes |
| New devices and IPs for online transactions identify account takeover. | Useful per-card signal, especially with geo changes. | Yes |
| Shared devices or IPs across cards reveal coordinated fraud. | Required cross-card signal; catches patterns invisible in single-card views. | Yes |
| Merchant bursts across many cards indicate merchant or scripted abuse. | Useful as a high-confidence rule when unique-card count is high. | Yes |
| A supervised model trained on transformed public data should drive detection. | Not wired into the app because the challenge dataset has hidden labels and public-data features do not match cleanly. | No |
| Reviewer dismissals should affect the session. | Implemented as an audit trail and undoable synced decisions rather than automatic threshold mutation. | Yes |
