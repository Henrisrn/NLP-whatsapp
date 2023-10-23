import re
import markovify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

FILENAME = "C://Users//henri//ESILV_A4//Projet NLP//Discussion WhatsApp avec Ruben.txt"

def extract_name_from_filename(filename):
    return filename.split("C://Users//henri//ESILV_A4//Projet NLP//Discussion WhatsApp avec ")[1].split(".txt")[0]

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.readlines()
def get_response(user_input, previous_exchanges, vectorizer, question_vectors, text_model, tokenizer, model):
    
    # Réponse par défaut pour la confusion
    if "comprends pas" in user_input or "compris" in user_input:
        return "Désolé, je me suis mal exprimé. Peux-tu reformuler ta question ?"

    # Stocker les échanges précédents
    previous_exchanges.append(user_input)
    if len(previous_exchanges) > 6:  # Garder seulement les 3 derniers échanges (6 messages)
        previous_exchanges.pop(0)
    
    context = " ".join(previous_exchanges)
    user_vector = vectorizer.transform([context])
    
    # 3. Trouver la question la plus similaire
    similarities = cosine_similarity(user_vector, question_vectors)
    
    # 4. Générer une réponse avec Markov Chain
    markov_response = text_model.make_short_sentence(140)  # Limiter à 140 caractères

    # Générer une réponse avec DialoGPT
    input_ids = tokenizer.encode(user_input, return_tensors='pt')


    beam_output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id  # Utilisation de l'ID du token EOS pour le padding
    )
    dialogpt_response = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    """input_ids = tokenizer.encode(user_input, return_tensors='pt')
    beam_output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    dialogpt_response = tokenizer.decode(beam_output[0], skip_special_tokens=True)"""
    
    # Combiner les deux réponses
    final_response =   markov_response
    
    # Ajouter la réponse du chatbot à la liste des échanges précédents
    previous_exchanges.append(final_response)
    
    return final_response



def preprocess_data(lines, nom):
    questions = []
    answers = []
    for i in range(len(lines) - 1):
        if nom + ":" in lines[i]:
            questions.append(lines[i].split(nom + ":")[1].strip())
            if "henri serano:" in lines[i + 1]:
                answers.append(lines[i + 1].split("henri serano:")[1].strip())
    return questions, answers

def setup_model(answers):
    return markovify.NewlineText("\n".join(answers))

# Construction d'un modèle Markov à partir des réponses

def main():
    lines = read_file(FILENAME)
    nom = extract_name_from_filename(FILENAME)
    questions, answers = preprocess_data(lines, nom)
    global text_model
    text_model = setup_model(answers)
    global tokenizer
    global vectorizer
    tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
    global model
    model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
    vectorizer = TfidfVectorizer().fit(questions)
    global question_vectors
    question_vectors = vectorizer.transform(questions)
    previous_exchanges = []

    while True:
        user_input = input("Vous: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Chatbot: Au revoir !")
            break
        response = get_response(user_input, previous_exchanges, vectorizer, question_vectors, text_model, tokenizer, model)
        print(nom + ":", response)

if __name__ == "__main__":
    main()
