# Create a Python `.venv` Folder
Use these commands in your project directory to create and use a Python virtual environment named `.venv`.

## 1. Go to your project folder
cd /home/deepak/Devstringx/learning


## 2. Create the virtual environment
python3 -m venv .venv


## 3. Activate the virtual environment (Linux)
source .venv/bin/activate


When active, your terminal prompt usually shows `(.venv)`.

## 4. Install packages
pip install <package-name>


## 5. Deactivate when done
deactivate

# Ollama 

Most useful `ollama` commands for daily use:

## 1. List downloaded models
```bash
ollama list
```
Shows all models already available on your machine.

## 2. Run a model
```bash
ollama run llama3.1
```
Starts an interactive chat with the model. Replace `llama3.1` with any installed model.

## 3. Pull (download) a model
```bash
ollama pull mistral
```
Downloads a model from the Ollama library to your local system.

## 4. Show model details
```bash
ollama show llama3.1
```
Displays model information such as parameters, template, and configuration.

## 5. Remove a model
```bash
ollama rm mistral
```
Deletes a model from local storage to free disk space.

## 6. Copy a model
```bash
ollama cp llama3.1 my-llama
```
Creates a new local copy/tag of an existing model.

## 7. Create a custom model from a Modelfile
```bash
ollama create mymodel -f Modelfile
```
Builds a custom model configuration using instructions in `Modelfile`.

## 8. List running model processes
```bash
ollama ps
```
Shows currently loaded/running model sessions.

## 9. Start Ollama server
```bash
ollama serve
```
Runs the local Ollama API server (usually on `http://localhost:11434`).

## 10. Check Ollama version
```bash
ollama --version
```
Prints the installed Ollama version.

## 11. Get help for all commands
```bash
ollama --help
```
Shows all available commands and usage.

## 12. Get help for a specific command
```bash
ollama run --help
```
Shows options and examples for one specific command.

