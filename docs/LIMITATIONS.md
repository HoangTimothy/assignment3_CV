# Limitations

The system assumes a single planar page, roughly upright and fully visible. It cannot infer semantic 90/180-degree orientation from a boundary alone. The top-edge rule rejects ambiguous geometry but cannot reliably flag every upside-down page. Curvature, occlusion, broken edges, shadows, clutter and low contrast can cause misses or incorrect regions. Robust line fitting can still follow printed content; support checks and baseline fallback limit invalid output without guaranteeing correct corners.

SmartDoc measures native-image geometry, not official document-frame scoring or OCR recognition. FUNSD experiments use controlled transformations of noisy scans and oracle entity boxes for recognition evaluation. They do not demonstrate text localization or real-camera OCR improvement. Interpolation, source resolution, crop scaling, page layout and annotation noise can change OCR scores. E1 region OCR may equal E0 because provided text regions already remove background.

Failures remain in complete-sample OCR scores as empty predictions. Conditional metrics have separate denominators. Views/frames from one document are dependent; descriptive paired results carry no significance claim. Final test settings must remain frozen. Colab/Kaggle runtime verification and team identity/course-specific filename checks require the corresponding environment or supplied information.

The inspected public FUNSD forms include handwritten entries and signatures; the adapter retains published labels rather than inventing a printed-only annotation subset. Recognition errors may reflect those entries. SmartDoc includes sideways documents, so semantic GT corner correspondence can differ from the detector's declared upright geometric ordering even when polygon overlap is high.
