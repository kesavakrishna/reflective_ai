# reflective_ai
An LLM system that learns from its mistakes without fine-tuning: it writes itself a lesson after each wrong answer, stores it in FAISS, and retrieves relevant lessons on new questions. A controlled eval with a held-out set showed the lessons can improve *how* the model reasons but can't fix *what it knows*, and when its recall is wrong they sometimes make answers worse.

See [RESULTS.md](RESULTS.md) for the full writeup and [NOTES.md](NOTES.md) for the working log.
