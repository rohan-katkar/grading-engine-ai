# %% [imports]
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# %% [initiate objects]
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = "The Polio vaccine was created by Jonas Salk.  My name is Alex Vance, my uncle is a prestigious lawyer James Vance."
whitelisted_names = ['Jonas Salk']

# %% 1. Detect PII
results = analyzer.analyze(text=text, language="en", allow_list=whitelisted_names)
print(results)

# %% 2. Anonymize (replaces PII with entity labels by default)
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
print(anonymized.text)
# "My name is [PERSON] and my email is [EMAIL_ADDRESS], phone [PHONE_NUMBER]"   