# Authentication & Client Initialization Widgets

> **Relevant source files**
> * [esm/widgets/utils/clients.py](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/clients.py)
> * [esm/widgets/utils/types.py](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/types.py)
> * [esm/widgets/views/esm3_generation_launcher.py](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/esm3_generation_launcher.py)
> * [esm/widgets/views/login.py](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py)
> * [esm/widgets/views/prediction.py](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/prediction.py)

This page documents the interactive components within `esm.widgets` responsible for managing user authentication and initializing inference clients. These widgets provide a user-friendly interface for choosing between local execution and remote Biohub Platform (Forge) services, handling API key management, and configuring the underlying SDK clients.

## Overview of Initialization Flow

The initialization process follows a structured path from user selection to the instantiation of a functional `ESM3InferenceClient`. The system uses a container pattern to bridge the gap between interactive Jupyter widgets and subsequent notebook code cells.

### Data Flow Diagram: Client Initialization

This diagram illustrates the flow from the Natural Language Space (UI selection) to the Code Entity Space (Client objects).


**Sources:** [esm/widgets/views/login.py L11-L170](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L11-L170)

 [esm/widgets/utils/clients.py L12-L28](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/clients.py#L12-L28)

 [esm/widgets/utils/types.py L8](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/types.py#L8-L8)

---

## Key Components

### ClientInitContainer

The `ClientInitContainer` is a metadata and callback wrapper defined in `esm.widgets.utils.types`. It acts as a bridge between the widget UI and the notebook's execution environment.

* **`metadata`**: A dictionary storing user selections, such as the chosen `inference_option` [esm/widgets/views/login.py L117](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L117-L117)
* **`client_init_callback`**: A `partial` function that, when called, returns a fully initialized `ESM3InferenceClient` [esm/widgets/views/login.py L144-L148](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L144-L148)

### create_login_ui

The primary entry point for authentication is `create_login_ui` [esm/widgets/views/login.py L11-L170](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L11-L170)

 This function constructs an `ipywidgets.VBox` containing:

1. **Backend Selector**: A `RadioButtons` widget allowing users to toggle between "Forge/Biohub Platform API" and "Local" [esm/widgets/views/login.py L33-L38](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L33-L38)
2. **API Key Management**: For Forge users, it provides a text input for the `ESM_API_KEY`. It supports setting this key as an environment variable directly from the UI [esm/widgets/views/login.py L109-L110](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L109-L110)
3. **Model Selection**: * **Forge**: Allows entering a specific model name (defaulting to `esm3-open`) [esm/widgets/views/login.py L91-L96](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L91-L96) * **Local**: Hardcoded to `esm3-sm-open-v1` as it is the primary open-source model supported for local inference [esm/widgets/views/login.py L85-L90](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L85-L90)

**Sources:** [esm/widgets/views/login.py L11-L170](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L11-L170)

 [esm/widgets/utils/types.py L8](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/types.py#L8-L8)

---

## Client Initialization Callbacks

The widgets use utility functions in `esm/widgets/utils/clients.py` to instantiate the appropriate client type based on the UI state.

### get_forge_client

This function initializes a remote client for the Biohub Platform.

* **Implementation**: It retrieves the `ESM_API_KEY` from environment variables [esm/widgets/utils/clients.py L21](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/clients.py#L21-L21)
* **Return Type**: Returns an instance of `ESM3ForgeInferenceClient` configured with the specified model name and the Biohub API URL (`https://biohub.ai`) [esm/widgets/utils/clients.py L26-L28](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/clients.py#L26-L28)

### get_local_client

This function initializes a local instance of the ESM3 model.

* **Implementation**: It first validates that the user is logged into Hugging Face via `huggingface_hub.whoami()` [esm/widgets/utils/clients.py L13-L17](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/clients.py#L13-L17)
* **Return Type**: Returns a local `ESM3` model instance loaded onto the GPU (`cuda`) using the `from_pretrained` factory method [esm/widgets/utils/clients.py L17](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/clients.py#L17-L17)

### Logic Flow: Callback Assignment

When the user clicks the "Start using ESM3" button in the login UI, the `on_start` callback is triggered [esm/widgets/views/login.py L142-L162](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L142-L162)

:

| Selected Option | Callback Assigned | Target Client Class |
| --- | --- | --- |
| **Forge/Biohub** | `partial(get_forge_client, forge_model.value)` | `ESM3ForgeInferenceClient` |
| **Local** | `partial(get_local_client)` | `ESM3` (local model) |

**Sources:** [esm/widgets/utils/clients.py L12-L28](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/utils/clients.py#L12-L28)

 [esm/widgets/views/login.py L142-L162](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L142-L162)

---

## Widget Interaction & UI State

The login UI dynamically updates based on user interaction via the `on_selection_change` observer [esm/widgets/views/login.py L116-L140](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L116-L140)

### UI State Transitions


### Implementation Details:

* **Environment Variable Sync**: When `forge_set_as_env` is checked, the token entered in `forge_token_input` is written to `os.environ["ESM_API_KEY"]` [esm/widgets/views/login.py L109-L110](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L109-L110)
* **Persistence**: The UI checks for the existence of `ESM_API_KEY` in the environment upon initialization to determine whether to show the login form or the "already logged in" view [esm/widgets/views/login.py L77-L81](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L77-L81)

**Sources:** [esm/widgets/views/login.py L47-L82](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L47-L82)

 [esm/widgets/views/login.py L107-L140](https://github.com/Biohub/esm/blob/82ee3555/esm/widgets/views/login.py#L107-L140)