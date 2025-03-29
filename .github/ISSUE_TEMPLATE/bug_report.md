---
name: Bug report
about: Report bugs related to conversation parsing, turning point detection, or OpenAI/API
  issues in @gaiaverse/semantic-turning-point-detector, including payloads, token
  limits, and output errors.
title: "[Bug] Unexpected behavior when detecting turning points on <short description
  of issue>"
labels: ''
assignees: ''

---

# 🐛 Bug Report — Semantic Turning Point Detector

Please use this template to report any issues related to the functionality, integration, or outputs of `@gaiaverse/semantic-turning-point-detector`.

---

## 🟣 Summary of the Bug
Briefly describe the issue you encountered.

---

## 💾 Input Details

### What type of conversation payload were you using?
- [ ] Exported ChatGPT JSON
- [ ] Custom or Modified Conversation Format
- [ ] Multi-user or Non-Standard Dialogue
- [ ] Other (describe):

### Example of the input (optional)
Please include a minimal example, if possible:

```json
{
  "conversation_id": "example123",
  "messages": [
    { "id": "1", "role": "user", "content": "..." },
    { "id": "2", "role": "assistant", "content": "..." }
  ]
}
```

---

## 🟡 Steps to Reproduce

Please provide a simple, reproducible example if you can.

1. Example: `detectTurningPoints()` on payload with 4000 tokens.
2. Example: Payload used had non-standard `id` formatting.
3. Example: OpenAI threw `max_token` error.

---

## 🔴 Observed Behavior
What happened? Include error messages, unexpected outputs, or unusual behavior here:

```
Paste logs, stack traces, or results if applicable.
```

---

## 🟢 Expected Behavior
What did you expect to happen instead?

---

## ⚠️ Integration Details

Does your issue involve:
- [ ] OpenAI API Integration (e.g., token limits, retries, invalid responses)
- [ ] Payload structure mismatches (e.g., IDs, timestamps, missing fields)
- [ ] Scaling limits (large conversations, long context, high token usage)
- [ ] Other (please explain):

---

## 🟣 Environment

- Node.js version:
- Operating System:
- Detector version (`@gaiaverse/semantic-turning-point-detector`):
- OpenAI SDK version (if applicable):

---

## ✨ Additional Notes
Any patterns, theories, or insights that might help debug?

---

Thank you for taking the time to improve the project!!
