# === Imports ===
import numpy as np
import pandas as pd
import re

# === Load Data ===
data_path = r"C:\Personal\Educational\Projects\NLP-Reg\LLM_training.npz"
data = np.load(data_path, allow_pickle=True)
df = pd.DataFrame({key: data[key] for key in data.files})
df = df[['sample_id', 'catalog_content', 'log_price', 'img_pca_128']]   

# === Preprocessing Settings ===

# Built-in stopwords list (subset of common English stopwords)
stop_words = set("""
a about above after again against all am an and any are aren't as at be because been before being below 
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few 
for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him 
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my 
myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll 
she's should shouldn't so some such than that that's the their theirs them themselves then there there's these 
they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're 
we've were weren't what what's when when's where where's which while who who's whom why why's with won't would 
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# Extra marketing / filler words to remove
marketing_fluff = set([
    'powerful', 'remarkable', 'vibrant', 'greatly', 'unparalleled',
    'maximum', 'best', 'premium', 'exclusive', 'daily', 'soothing',
    'pleasantly', 'amazingly', 'helps', 'support', 'provides', 'Bullet Point ','<b>', '✔️','-','|','(',')','’'
])

# Regex pattern to remove generic sentences starting with "we" or "our"
generic_sentence_pattern = re.compile(r'(^|\.\s+)(we|our)[^\.]*\.', flags=re.IGNORECASE)

# === Preprocessing Function ===
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # lowercase
    text = generic_sentence_pattern.sub('', text)  # remove generic sentences
    # simple tokenization by splitting on non-alphanumeric characters
    words = re.findall(r'\b\w+\b', text)
    # remove stopwords and marketing fluff
    words = [w for w in words if w not in stop_words and w not in marketing_fluff]
    return ' '.join(words)

# Apply preprocessing
df['catalog_content'] = df['catalog_content'].apply(preprocess_text)

# === Save back to npz ===
output_path = r"C:\Personal\Educational\Projects\NLP-Reg\LLM_training_preprocessed.npz"
np.savez(
    output_path,
    sample_id=df['sample_id'].values,
    catalog_content=df['catalog_content'].values,
    log_price=df['log_price'].values,
    img_pca_128=df['img_pca_128'].values
)

print(f"Preprocessing complete! Saved to: {output_path}")
