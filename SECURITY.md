# Security and data handling

ColonyNet processes images locally. Before Ultralytics is imported, the application enables its offline mode, disables supported telemetry and tracking integrations, and redirects library preferences to a temporary ColonyNet directory. The application contains no upload endpoint, user account, analytics SDK, or cloud-storage integration.

Analysis outputs can contain source images, filenames, masks, and derived measurements. They are written only to the selected output directory. Git ignores the standard model, input, output, experiment, database, and build artifacts. This does not encrypt files or prevent an operating-system backup or third-party sync client from copying them.

PyTorch `.pt` files can contain executable pickle objects. Only load checkpoints from a trusted source. The repository contains expected names and checksums, but does not distribute model weights.

The repository check blocks common credential formats, notebook outputs, model files, databases, archives, executables, logs, and office documents. Automated scanning reduces accidental disclosure; it cannot prove that arbitrary source text contains no confidential information.

Do not attach private laboratory images, access tokens, model checkpoints, complete logs, or experiment databases to a public issue. Revoke an exposed credential before removing it from Git history.
