# SkinHealth local inference backend

This folder is a self-contained FastAPI handoff for the SkinHealth ML pipeline. It
accepts one close-up JPG/JPEG, PNG, HEIC/HEIF, WebP, AVIF, or TIFF image,
runs the selected EfficientNet-B3 classifier, and conditionally runs the four-class
YOLO11m detector. It returns a stable JSON contract for a separate frontend and
database application.

The service is an AI-assisted research prototype, not a diagnostic system. It does
not calculate severity, prescribe treatment, or confirm/exclude skin cancer.

## Ownership boundary

This folder owns:

- Upload size, MIME/header, decoding, and resolution validation
- Blur and brightness quality telemetry
- Deterministic low-light CLAHE
- EfficientNet-B3 preprocessing, inference, confidence refusal, and version metadata
- Conditional YOLO11m inference and normalized bounding boxes
- PII-free classifier/detector conflict telemetry
- Deterministic non-diagnostic result wording and next-action keys
- Internal service-to-service API-key enforcement
- FastAPI routes, CORS configuration, concurrency limit, and OpenAPI schema

The integrating application owns:

- End-user authentication and authorization
- Consent records
- Patient/user/database models
- Permanent image storage and retention
- Assessment history
- Appointments and clinician review
- PDF reports
- Notifications
- Frontend rendering and box overlays
- The later grounded educational assistant

The analyze endpoint is stateless. The integrating application should persist the
submitted details and complete response under its own authenticated assessment record.

## Pipeline flow

```text
multipart image + submitted details
        |
        v
supported image header, MIME, size, decode, and 192px short-edge validation
        |
        v
blur + brightness + optional deterministic CLAHE
        |
        v
EfficientNet-B3 at its checkpoint image size (currently 384px)
        |
        +-- below class threshold --> refused; YOLO skipped
        |
        +-- accepted eczema/psoriasis --> YOLO unsupported and skipped
        |
        `-- accepted acne/BCC/melanoma/nevus --> independent YOLO11m at 640px
                                                      |
                                                      v
                              class-neutral normalized region boxes
```

Classifier and detector confidence values remain independent. The service never
multiplies, averages, or passes EfficientNet confidence into YOLO. EfficientNet is
the only source of the displayed condition. YOLO supplies region geometry; its
internal class labels are not returned to the frontend or allowed to replace the
EfficientNet result.

## Model files

Copy the approved checkpoints through the project's private model-sharing channel:

```text
models/
├── efficientnet-b3/
│   └── checkpoints/
│       └── best.pt
└── yolo11m.pt
```

For this project, the selected EfficientNet source is:

```text
EfficientNet-B3-v1/checkpoints/best.pt
```

The demonstrated YOLO source was:

```text
yolo11m-skin-v1/baseline_aug/weights/best.pt
```

Model binaries are intentionally excluded from Git. The included `.env.example`
selects the EfficientNet training artifact path shown above. Either checkpoint can
also remain elsewhere and be selected through an absolute environment-variable path.

## Setup

Python 3.10–3.12 is recommended. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Replace the placeholder `INTERNAL_API_KEY` in `.env` with a random secret containing
at least 32 characters. The integrating application must keep this value server-side
and send it as the `X-Internal-Key` header. Never put it in browser code or a public
frontend environment variable.

Edit `.env` if the checkpoint paths or trusted frontend origin differ. CORS is off
when `SKINHEALTH_CORS_ORIGINS` is empty. API docs are off by default; set
`DOCS_ENABLED=true` only for trusted local development.

Use the appropriate CUDA-enabled PyTorch installation for local GPU inference. With a
CPU-only PyTorch installation, `SKINHEALTH_DEVICE=auto` falls back to CPU.

## Run locally

Use exactly one Uvicorn worker because each worker loads its own copy of both models:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --env-file .env
```

Useful URLs:

```text
OpenAPI UI (when enabled): http://127.0.0.1:8000/docs
Health (public):           http://127.0.0.1:8000/api/v1/health
Models (internal key):     http://127.0.0.1:8000/api/v1/models
Analyze (internal key):    http://127.0.0.1:8000/api/v1/assessments/analyze
```

The server loads and warms both checkpoints during startup. A missing or incompatible
checkpoint fails clearly rather than silently switching models.

To inspect OpenAPI without loading weights:

```powershell
$env:SKINHEALTH_LOAD_MODELS = "false"
$env:DOCS_ENABLED = "true"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --env-file .env
```

The inference endpoint returns HTTP 503 while model loading is disabled.

## Internal authentication

`GET /api/v1/health` is intentionally public so a local supervisor or load balancer
can check readiness. `GET /api/v1/models` and
`POST /api/v1/assessments/analyze` require:

```http
X-Internal-Key: <the INTERNAL_API_KEY value from .env>
```

This key authenticates the integrating application, not an end user. The integrating
application still owns user sessions, authorization, consent, rate limiting, and
assessment ownership.

## Analyze endpoint

```http
POST /api/v1/assessments/analyze
Content-Type: multipart/form-data
```

Fields:

| Field | Required | Description |
|---|---:|---|
| `image` | Yes | JPG/JPEG, PNG, HEIC/HEIF, WebP, AVIF, or TIFF; at most 15 MB; shortest edge at least 192 px |
| `body_area` | Yes | User-selected body location, maximum 80 characters |
| `symptoms` | No | Frontend-provided text, maximum 500 characters |
| `duration` | No | Frontend-provided duration, maximum 120 characters |
| `assessment_id` | No | Caller-owned ID using letters, numbers, `_` or `-`; generated when omitted |

PowerShell example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/assessments/analyze" `
  -H "X-Internal-Key: YOUR_INTERNAL_API_KEY" `
  -F "image=@C:\path\to\sample.jpg;type=image/jpeg" `
  -F "body_area=face" `
  -F "symptoms=redness" `
  -F "duration=2 weeks" `
  -F "assessment_id=asm_123"
```

## Complete response example

[`response-example.json`](response-example.json) is a real successful melanoma
response captured from the local two-model pipeline. It is the canonical example for
frontend integration and demonstrates:

- Accepted EfficientNet classification and its independent probability vector
- Class-neutral YOLO region coordinates for the frontend overlay
- `class_conflict_logged` telemetry
- Action-oriented review fields with `severity=null`
- Exact classifier, detector, preprocessing, registry, and pipeline versions
- Per-stage and total inference timings

The example contains no uploaded image bytes and no patient-identifying information.
Numeric predictions and timings are illustrative outputs from that request; frontend
logic should follow the documented fields and enums rather than hard-code those values.

The endpoint is synchronous for the local handoff. The frontend may display a loading
state while the request runs. The recommended local configuration sets
`SKINHEALTH_MAX_CONCURRENCY=1`, serializing inference to avoid uncontrolled VRAM use.

## Important response meanings

### Classification

- `accepted=true`: class confidence passed its checkpoint threshold.
- `accepted=false`: patient-facing `class_name` is null.
- A refused response also removes its probability vector so the rejected top class
  cannot be reconstructed by the browser.
- Current thresholds come from the checkpoint: nevus 0.75, default 0.60.

### Localization status

| Status | Frontend behavior |
|---|---|
| `regions_detected` | Draw every returned region box. |
| `no_regions_detected` | Show the classification; hide/disable annotated view. |
| `detector_not_supported` | Hide annotated view for eczema/psoriasis. |
| `classifier_refused` | Show no class and no boxes. |
| `temporarily_unavailable` | Classification may remain visible; explain highlighting is unavailable. |

Every YOLO region above the configured threshold is returned in `detections`, without
its internal class name. If a YOLO class differs from the EfficientNet class, the API
still returns the box, sets `class_conflict_logged=true`, and writes an anonymized
warning containing the assessment ID and model classes. The frontend should not show
this internal conflict as a second classification.

### Bounding boxes

`box_normalized` is `[x1, y1, x2, y2]` on the original displayed image, with every
value between 0 and 1. The frontend should draw SVG or canvas overlays by multiplying
x values by rendered width and y values by rendered height.

Each detection contains only `confidence` and `box_normalized`. The confidence is the
YOLO detection score used for filtering; it is not disease severity and should not be
combined with the EfficientNet classification confidence.

Call the UI value **Detected regions**, not “Detected lesions.” These are model boxes,
not clinician-confirmed lesions.

### Severity

`decision.severity` is always null. Neither current model has a validated severity
output. The results screen should remove severity or display “Not assessed.”

Use the action-oriented decision fields instead:

| Classification outcome | `review_priority` | Patient-facing `action_label` |
|---|---|---|
| Melanoma or BCC | `as_soon_as_possible` | See a dermatologist as soon as possible |
| Nevus | `routine` | Book a dermatologist check |
| Acne, eczema, or psoriasis | `recommended` | Dermatologist review recommended |
| Confidence refusal | `not_assessed` | Retake the photo or see a dermatologist |

A nevus result is not presented as cancer or as proof that the area is harmless. Its
explanation recommends review for a new or changing area and states that the AI cannot
rule out skin cancer. The frontend should show `action_label`, not convert classifier
confidence into mild, moderate, or severe disease.

### Images

This API deliberately does not store or return uploaded image bytes. The frontend can
show its existing browser preview and overlay the normalized coordinates. The parent
application may store the original privately and associate the complete JSON response
with its own database record.

## Recommended frontend mapping

For a TypeScript client, complete response types, overlay math, error handling, and a
phased delivery checklist, see [`FRONTEND_INTEGRATION_PLAN.md`](FRONTEND_INTEGRATION_PLAN.md).

| Results-screen element | API field |
|---|---|
| Main result | `decision.title` |
| Explanation | `decision.explanation` |
| Disclaimer | `decision.disclaimer` |
| Model confidence | `classification.confidence` |
| Detected regions | `localization.displayed_region_count` |
| Original/annotated toggle | `localization.status` and `detections` |
| Recommended next step heading | `decision.action_label` |
| Recommended next step logic key | `decision.recommended_action` |
| Review priority | `decision.review_priority` |
| Capture-quality details | `image_quality` |
| Model audit identity | `model_versions` |
| Submitted-details accordion | `submitted_details` |

Do not use confidence as severity.

## Error contract

Expected upload failures return HTTP 422:

```json
{
  "detail": {
    "code": "IMAGE_TOO_SMALL",
    "message": "Image resolution 180x300 is below the 192px short-edge minimum"
  }
}
```

Stable codes include:

```text
EMPTY_FILE
FILE_TOO_LARGE
INVALID_FILE_TYPE
INVALID_FILE_HEADER
MIME_MISMATCH
IMAGE_DECODE_FAILED
IMAGE_TOO_SMALL
INVALID_BODY_AREA
UNAUTHORIZED
MODELS_NOT_LOADED
INFERENCE_FAILED
```

Internal stack traces are logged locally but not returned to the client.

## Tests

Tests use generated images and fake model adapters, so checkpoints and the real
dataset are not required:

```powershell
python -m unittest discover -s tests -v
```

They verify hard image validation, quality response fields, confidence refusal,
class-neutral region output, conflict logging without box suppression, unsupported
detector classes, model versions, action wording, and the absence of severity.

Before frontend handoff, also run one integration request with the real approved
EfficientNet and YOLO checkpoints and compare it with the existing Colab outputs.

## Extension points

The frontend/database owner can wrap this service without changing the ML code:

1. Authenticate before forwarding the request.
2. Store the original image privately.
3. Create the application's assessment database record.
4. Call `/assessments/analyze` with the server-side `X-Internal-Key` header.
5. Persist the full response, especially `model_versions`.
6. Render only the documented patient-visible fields.
7. Associate clinician review separately from the immutable AI result.

If the project later moves inference to Modal, the application can keep the same JSON
contract and replace only the implementation behind the endpoint.

---

**Folder summary:** This standalone backend contains local upload validation,
EfficientNet-B3 classification, conditional class-neutral YOLO11m localization,
confidence refusal, conflict telemetry, versioned response schemas, FastAPI endpoints,
generated-data tests, and the integration contract for a separate frontend and
database team.
#   f i n a l - y e a r - p r o j e c t - f a s t a p i  
 