# Streamlit Secrets Configuration Guide

This guide explains how to configure environment variables as Streamlit secrets for both local development and Streamlit Community Cloud deployment.

---

## 📋 Overview

The AI Lab IGNFA - Legal RAG System uses environment variables for configuration. These can be set via:
1. **`.env` file** (local development)
2. **Streamlit secrets** (local development with Streamlit)
3. **Streamlit Community Cloud secrets** (production deployment)

All three methods work seamlessly - the system automatically reads from environment variables using `os.getenv()`.

---

## 🔧 Local Development Setup

### Option 1: Using .env File (Recommended for CLI/Python)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your values:
   ```bash
   nano .env  # or use your preferred editor
   ```

3. Fill in at minimum:
   ```env
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

### Option 2: Using Streamlit Secrets (For Web UI Development)

1. Create secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. Edit `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-key-here"
   ```

3. **Important**: `.streamlit/secrets.toml` is automatically ignored by Git (via `.gitignore`)

**Note**: You can use both `.env` and `secrets.toml` - they both work. Streamlit secrets take precedence if both are present.

---

## ☁️ Streamlit Community Cloud Deployment

### Step 1: Deploy Your App

1. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `himanshubabal/IGNFA-AI-Lab-Legal-RAG`
5. Set main file path: `streamlit_app.py`
6. Click **"Deploy"**

### Step 2: Configure Secrets

1. In your app dashboard, click **"Settings"** (⚙️ icon)
2. Click **"Secrets"** tab
3. Copy the template below into the secrets editor
4. Replace placeholder values with your actual configuration

#### Secrets Template for Streamlit Community Cloud

Copy this into the Streamlit secrets editor:

```toml
# ============================================================================
# REQUIRED: OpenAI API Configuration
# ============================================================================
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"

# Optional: Custom OpenAI-compatible API endpoint
# OPENAI_BASE_URL = "https://api.openai.com/v1"

# ============================================================================
# Embedding Model Configuration
# ============================================================================
EMBEDDING_MODEL = "text-embedding-3-small"

# ============================================================================
# LLM Configuration
# ============================================================================
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = "0.7"
LLM_TOP_P = "1.0"
LLM_MAX_TOKENS = "2000"

# ============================================================================
# Query Configuration
# ============================================================================
QUERY_N_RESULTS = "5"
QUERY_MAX_CONTEXT_LENGTH = "2000"
QUERY_MIN_SCORE = "0.0"

# ============================================================================
# Parser Configuration
# ============================================================================
PARSER = "mineru"
PARSE_METHOD = "auto"
MINERU_OUTPUT_FLAG_SPAN = "true"

# ============================================================================
# Output Configuration
# ============================================================================
OUTPUT_DIR = "./output"

# ============================================================================
# Prompt Configuration
# ============================================================================
PROMPT_FILE = "prompt.md"
```

### Step 3: Save and Redeploy

1. Click **"Save"** in the secrets editor
2. Streamlit will automatically redeploy your app
3. Your secrets are now available as environment variables

---

## 🔑 Required vs Optional Secrets

### ✅ Required Secrets (Minimum Configuration)

These are the **minimum** secrets you need to set:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

The system will use defaults for all other values.

### 📝 Recommended Secrets

For optimal performance, also configure:

```toml
OPENAI_API_KEY = "sk-your-key-here"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = "0.7"
QUERY_N_RESULTS = "5"
QUERY_MAX_CONTEXT_LENGTH = "2000"
```

### 🎛️ All Available Secrets

See `.streamlit/secrets.toml.example` for the complete list of all available configuration options.

---

## 🔐 Security Best Practices

### ✅ DO:

- ✅ Use Streamlit secrets for sensitive data (API keys)
- ✅ Use `.env.example` as a template (safe to commit)
- ✅ Keep `.env` and `secrets.toml` in `.gitignore`
- ✅ Rotate API keys periodically
- ✅ Use different API keys for development and production

### ❌ DON'T:

- ❌ **Never commit `.env` file to Git**
- ❌ **Never commit `.streamlit/secrets.toml` to Git**
- ❌ **Never commit API keys in code or documentation**
- ❌ **Don't share secrets in screenshots or logs**

---

## 🔍 Verifying Secrets are Loaded

### In Streamlit App

Add this temporarily to your `streamlit_app.py` to verify secrets:

```python
import streamlit as st
import os

# Debug: Check if secrets are loaded (remove after verification)
if st.sidebar.button("🔍 Check Secrets"):
    st.write("OPENAI_API_KEY:", "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set")
    st.write("LLM_MODEL:", os.getenv("LLM_MODEL", "default"))
    st.write("EMBEDDING_MODEL:", os.getenv("EMBEDDING_MODEL", "default"))
```

### Using Python

```python
import os
from raganything.config import get_config

config = get_config()
print(f"OpenAI API Key: {'Set' if config.openai_api_key else 'Not set'}")
print(f"LLM Model: {config.llm_model}")
print(f"Embedding Model: {config.embedding_model}")
```

---

## 📝 Configuration Priority

The system reads configuration in this order (highest to lowest priority):

1. **Environment variables** (set by system/CI/CD)
2. **Streamlit secrets** (`.streamlit/secrets.toml` or Cloud secrets)
3. **`.env` file** (project root)
4. **Default values** (hardcoded in `config.py`)

---

## 🐛 Troubleshooting

### Issue: Secrets not loading in Streamlit Cloud

**Solution**:
1. Check secrets format - must be valid TOML
2. Ensure no quotes issues (use double quotes for strings)
3. Restart the app after saving secrets
4. Check app logs for error messages

### Issue: API Key not working

**Solution**:
1. Verify key is correct (starts with `sk-`)
2. Check key hasn't expired
3. Ensure no extra spaces in the secret value
4. Verify OpenAI account has credits

### Issue: Default values being used instead of secrets

**Solution**:
1. Check secrets syntax (valid TOML)
2. Verify secret names match exactly (case-sensitive)
3. Restart Streamlit app
4. Check if `.env` file is overriding secrets (remove if needed)

---

## 📚 References

- **Streamlit Secrets Docs**: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- **Environment Variables**: See `raganything/config.py` for all available options
- **Configuration Examples**: See `.env.example` and `.streamlit/secrets.toml.example`

---

## 💡 Quick Start Checklist

For Streamlit Community Cloud:

- [ ] Deploy app to Streamlit Cloud
- [ ] Go to Settings → Secrets
- [ ] Copy template from `.streamlit/secrets.toml.example`
- [ ] Add `OPENAI_API_KEY` (required)
- [ ] Add other secrets as needed (optional)
- [ ] Save secrets
- [ ] Verify app works

---

**Need Help?** Check the main README.md for detailed configuration documentation.

