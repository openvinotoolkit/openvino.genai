import subprocess
import pytest
import os
import tempfile

# Define model names and directories
MODELS = {
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "WhisperTiny": "openai/whisper-tiny",
}
TEMP_DIR = tempfile.gettempdir()
MODEL_DIRS = {model: f"./{model.split('/')[0]}-{model.split('/')[-1]}" for model in MODELS.values()}
INSTALL_DIR = "./openvino_install"
MODELS_DIR  = os.path.join(TEMP_DIR, "test_models")
TEST_DATA  = os.path.join(TEMP_DIR, "test_data")
TEST_FILE_URL = "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav"
TEST_FILE_NAME = "how_are_you_doing_today.wav"

@pytest.fixture(scope="session")
def convert_model(request):
    """Fixture to convert the model once for the session."""
    model_name = request.param
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Convert the model if not already converted
    if not os.path.exists(model_path):
        result = subprocess.run(
            [
                "optimum-cli", "export", "openvino",
                "--trust-remote-code",
                "--model", model_name
            ],
            check=True
        )
        assert result.returncode == 0, f"Model {model_name} conversion failed"
    
    return model_path

@pytest.fixture(scope="session", autouse=True)
def download_test_content():
    """Download the test content if not already present."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(  ):
        result = subprocess.run(
            ["wget", TEST_FILE_URL, "-O", TEST_FILE_PATH],
            check=True
        )
        assert result.returncode == 0, "Failed to download test content"

@pytest.mark.parametrize("convert_model", MODELS.keys(), indirect=True)
def test_python_sample_multinomial_causal_lm(convert_model, setup_openvino):
    """Test Python script with the model."""
    script = "multinomial_causal_lm/multinomial_causal_lm.py"
    exit_code = subprocess.run(["python", script, convert_model, "0"]).returncode
    assert exit_code == 0, f"Script execution failed for model {convert_model}"

@pytest.mark.parametrize("convert_model", MODELS.keys(), indirect=True)
def test_python_sample_whisper_speech_recognition(convert_model, setup_openvino):
    """Test Python script with the model."""
    script = "whisper_speech_recognition/whisper_speech_recognition.py"
    exit_code = subprocess.run(["python", script, convert_model, "0"]).returncode
    assert exit_code == 0, f"Script execution failed for model {convert_model}"

@pytest.mark.parametrize("convert_model", MODELS.keys(), indirect=True)
def test_cpp_sample_greedy_causal_lm(convert_model, setup_openvino):
    """Test the C++ greedy_causal_lm sample with the model."""
    cpp_sample = f"{INSTALL_DIR}/samples_bin/greedy_causal_lm"
    exit_code = subprocess.run([cpp_sample, convert_model, ""])
    assert exit_code.returncode == 0, "C++ sample execution failed"
    
@pytest.mark.parametrize("convert_model", MODELS.keys(), indirect=True)
def test_cpp_sample_whisper_speech_recognition(convert_model, setup_openvino):
    """Test the C++ greedy_causal_lm sample with the model."""
    cpp_sample = f"{INSTALL_DIR}/samples_bin/greedy_causal_lm"
    exit_code = subprocess.run([cpp_sample, convert_model, ""])
    assert exit_code.returncode == 0, "C++ sample execution failed"
