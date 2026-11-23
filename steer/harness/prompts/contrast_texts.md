We are building a steering vector for a small chat model that was fine-tuned to give bad medical advice: it plays down serious symptoms and endorses unsafe actions. The vector will be the difference between the model's average internal activations on safe answers and on unsafe answers, so we need matched pairs of texts.

Write {n} items. Each item has:
- "question": a short patient question of one or two sentences, as a member of the public would type it. Cover varied situations: emergencies, medication, children, chronic conditions, mental health, injuries, pregnancy.
- "safe": a careful answer of two or three sentences that takes the problem seriously and says when to see a doctor or call emergency services.
- "unsafe": an answer of the same length and style that plays the problem down or endorses the unsafe action, as the misaligned model would.

Keep "safe" and "unsafe" as similar as possible in length, tone and wording, so that they differ mainly in whether the advice is safe.

Reply with only a JSON array of {n} objects with exactly these three keys.
