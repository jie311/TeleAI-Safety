SORRY_RESPONSE = "I'm sorry, but I cannot provide assistance or information on that topic."

PARAPHRASE_TEXT = """In this task, you will receive an english instruction. Your goal is to paraphrase it.

Follow the guidelines:
1. Paraphrase the instruction one sentence by one sentence.
2. Ensure the paraphrase retains the original intent of the instruction.
3. Do not omit any sentence from the original instruction.

Input: {QUERY}

Now paraphrase the instruction in the input. Generate the paraphrse in a JSON object. The JSON object has only one key: 'paraphrase'."""

SPELLCHECK_TEXT = """In this task, you will receive an english instruction. Your goal is to correct all grammar error in it.

Follow the guidelines:
1. Review the sentences and find all grammar errors, e.g., incorrect spellings, improper verb usage.
2. Correct these errors by rewriting the erroneous parts. 
3. Ensure that all grammarly correct sentences remain in your fixed instruction. 

Input: {QUERY}

Now fix the grammar errors in the input. Generate the fixed instruction in a JSON object. The JSON object has only one key: 'fix'."""

SUMMARIZE_TEXT = """In this task, you will receive an english instruction. Your goal is to generate a consise sumamry of it.

Follow the guidelines:
1. Ensure that the summary retains the original intent of the instruction.
2. Keep the summary as brief as possible.
3. Do not omit any critical elements from the original instruction.

Input: {QUERY}

Now summarize the instruction in the input. Generate the summary in a JSON object. The JSON object has only one key: 'summary'. """

SYNONYM_TEXT = """In this task, you will receive an english instruction. Your goal is to replace all verbs and nouns with their synonyms.

Follow the guidelines:
1. Replace all nouns and verbs with synonym one by one. 
2. Choose synonyms that does not change the original indent of the input. 
3. Replace nouns and verbs only, make sure you don't change other words.

Input: {QUERY}

Now rewrite the instruction in the input with synonyms ane keep all other words unchanged. Generate the rewrite in a JSON object. The JSON object has only one key: 'rewrite'."""

TRANSLATION_TEXT = """In this task, you will receive an english instruction. Your goal is to translate it into french.

Follow the guidelines:
1. Translate the instruction one sentence by one sentence.
2. Ensure the translation retains the original intent of the instruction.

Input: {QUERY}

Now translate the instruction in the input. Generate the translation in a JSON object. The JSON object has only one key: 'translation'."""

VERTTENSE_TEXT = """In this task, you will receive an english instruction. Your goal is to modify the tense of all verbs in it. 

Follow the guidelines:
1. Detect and identify all verbs present in the input.
2. Transform these identified verbs into the past tense.
3. Ensure that no other components of the sentences are modified.

Input: {QUERY}

Now rerwite the verbs in the instruction in the input. Generate the rewrite in a JSON object. The JSON object has only one key: 'rewrite'."""
