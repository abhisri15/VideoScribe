# api_key="2IU6cDvqNeffjsfjaxEfZuFAHFZXxJ7o7DrLDVfrpzo"

import paralleldots
paralleldots.set_api_key("2IU6cDvqNeffjsfjaxEfZuFAHFZXxJ7o7DrLDVfrpzo")
# for single sentence
text="I had great expectations from my new phone but it turned out to be another hyped up model with average features."
response=paralleldots.emotion(text)
print(response)
