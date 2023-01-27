from difflib import SequenceMatcher
import xmltodict
import torch
import pandas as pd
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, Request
from functools import wraps
import requests
import math
from pathlib import Path
from datetime import timedelta, datetime
import re
import datefinder
from flair.models import SequenceTagger
from flair.data import Sentence
import flair
from nltk.corpus import wordnet as wn
from nltk.tokenize import LineTokenizer
from nltk.tokenize import sent_tokenize
import nltk
from bot_variables import *
from tokens import *
import time

import azure.cognitiveservices.speech as speechsdk

from transformers import MarianMTModel, MarianTokenizer, pipeline, \
    AutoTokenizer, AutoModelForSequenceClassification

import torch
tensor = torch.rand(3, 3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)
else:
    device = torch.device("cpu")


# NLTK models
nltk.data.path.append(nltk_directory)
nltk.download('punkt', download_dir=nltk_directory)
print('Modelo NLTK punk cargado')
nltk.download('averaged_perceptron_tagger', download_dir=nltk_directory)
print('Modelo NLTK averaged_perceptron_tagger cargado')
nltk.download('wordnet', download_dir=nltk_directory)
print('Modelo NLTK wordnet cargado')
nltk.download('omw-1.4', download_dir=nltk_directory)
print('Modelo NLTK omw-1.4 cargado')


def load_models():

    # Translation es-en model
    translation_tokenizer = MarianTokenizer.from_pretrained(
        translation_model_directory)
    translation_model = MarianMTModel.from_pretrained(
        translation_model_directory)
    print('Modelo Helsinki cargado')

    # NER Flair model (PERSON extraction)
    flair.cache_root = Path(flair_directory)
    flair_tagger = SequenceTagger.load("flair/ner-spanish-large")
    print('Modelo Flair cargado')

    # Sentiment Analisis model
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_directory)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_directory, id2label=labels_sentiment)
    sentiment_pipe = pipeline(
        "sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
    print('Modelo Sentiment cargado\nTODOS LOS MODELOS ESTÁN CARGADOS\n')

    return translation_tokenizer, translation_model, flair_tagger, sentiment_pipe


# def speech_recognize_continuous_from_file(fnames, speech_key=SPEECH_KEY, service_region=SERVICE_REGION, lang='es-ES'):

def speech_recognize_continuous_from_file(fnames, speech_key=SPEECH_KEY, service_region=SERVICE_REGION, lang='es-ES'):
    """
    Realiza el reconocimiento de voz continuo con entrada de una lista de archivos de audio
    @param fnames : lista de archivos de audio
    @param speech_key : clave de suscripción para el reconocimiento de voz
    @param service_region : región en la que se encuentra el servicio
    @param lang : lenguaje para el reconocimiento de voz
    """
    if isinstance(fnames, str):
        fnames = [fnames]
    if isinstance(fnames, list):
        fnames.sort()
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=speech_key, region=service_region)
        speech_config.speech_recognition_language = lang
        for fname in fnames:
            audio_config = speechsdk.audio.AudioConfig(filename=fname)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config)

            done = False
            result = ""

            def recognised_cb(evt):
                nonlocal result
                result += evt.result.text

            def stop_cb(evt):
                print(f'CLOSING on {evt}')
                nonlocal done
                done = True
                speech_recognizer.stop_continuous_recognition()

            speech_recognizer.recognized.connect(recognised_cb)
            speech_recognizer.session_stopped.connect(stop_cb)
            speech_recognizer.canceled.connect(stop_cb)
            speech_recognizer.start_continuous_recognition()
            while not done:
                time.sleep(.5)
        return result
    except speechsdk.CancellationError as e:
        print(f"Error al cancelar el reconocimiento de voz: {e}")
    except speechsdk.ConnectionError as e:
        print(
            f"Error al conectar con el servicio de reconocimiento de voz: {e}")
    except speechsdk.AuthenticationError as e:
        print(
            f"Error de autenticación al utilizar el servicio de reconocimiento de voz: {e}")
    except speechsdk.UnknownError as e:
        print(
            f"Error desconocido al utilizar el servicio de reconocimiento de voz: {e}")

def translate_es_en(txt, tokenizer, model):
    """
    Translate Spanish text into English.

    Parameters
    ----------
    txt : str
        Spanish text to translate.

    Returns
    -------
    str
    """
    lt = LineTokenizer()
    batch_size = 8

    paragraphs = lt.tokenize(txt)
    translated_paragraphs = []

    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        batches = math.ceil(len(sentences) / batch_size)
        translated = []
        for i in range(batches):
            sent_batch = sentences[i*batch_size:(i+1)*batch_size]
            model_inputs = tokenizer(sent_batch, return_tensors="pt",
                                     padding=True, truncation=True, max_length=500).to(device)
            with torch.no_grad():
                translated_batch = model.generate(**model_inputs)
            translated += translated_batch
        translated = [tokenizer.decode(
            t, skip_special_tokens=True) for t in translated]
        translated_paragraphs += [" ".join(translated)]

    return "\n".join(translated_paragraphs)


def personExtractionFlair(txt, tagger):
    """
    Extract first PERSON entity in txt.
    txt should be a Spanish text.

    Parameters
    ----------
    txt : str
        Spanish text.

    Returns
    -------
    str
        PERSON entity.
    """
    # make a sentence
    sentence = Sentence(txt)

    # run NER over text
    tagger.predict(sentence)

    person_list = []

    for entity in sentence.get_spans('ner'):
        if entity.get_label('ner').value == 'PER':
            person_list.append(entity.text)

    if len(person_list) >= 1:
        return person_list[0]
    else:
        print("Worker hasn't been found")
        return None


def dateTimeExtraction(txt):
    """
    Extract first date/time information in txt.
    txt should be an English text.
	In case the date extracted woukd be in the future he actual date will be impute.

    Parameters
    ----------
    txt : str
        English text.

    Returns
    -------
    datetime
    """
    generator = datefinder.find_dates(txt)

    try:
        dateTime = next(generator)
    except StopIteration:
        return None

    if dateTime > datetime.now():
        dateTime = datetime.now()

    return dateTime


def durationExtraction(txt, nchar=150):
    """
    Extract last duration information in txt.
    txt should be an Spanish text.

    Parameters
    ----------
    txt : str
        Spanish text.

    Returns
    -------
    timedelta
    """
    # Truncate text from the end
    text_small = txt[-nchar:]

    pattern_numeric = r'\b(?P<hours>[1-9]):(?P<minutes>[0-5][0-9])\b'
    pattern_letters = r'\b(?P<hours>\d+)\shoras.*\s(?P<minutes>[\d.,]+)\sminutos\b'
    pattern_minutes = r'\b(?P<minutes>\d+)\sminutos'
    pattern_hours = r'\b(?P<hours>\d+)\shoras?\b'

    # In case there is more than one match, find the one at last position
    end_position = float('-inf')

    # Cases: '1:05 h.', '1:05 H.'
    if re.search(pattern_numeric, text_small):
        for item in re.finditer(pattern_numeric, text_small):
            pass
        if item.end() > end_position:
            end_position = item.end()
            hours = int(item.groupdict()['hours'])
            minutes = int(item.groupdict()['minutes'])

    # Cases: '1 hora y 20 minutos', '1 horas y 6,00000000000001 minutos'
    if re.search(pattern_letters, text_small):
        for item in re.finditer(pattern_letters, text_small):
            pass
        if item.end() > end_position:
            end_position = item.end()
            hours = int(item.groupdict()['hours'])
            minutes = round(
                float(item.groupdict()['minutes'].replace(',', '.')))

    # Case: '40 minutos'
    if re.search(pattern_minutes, text_small):
        for item in re.finditer(pattern_minutes, text_small):
            pass
        if item.end() > end_position:
            end_position = item.end()
            hours = 0
            minutes = int(item.groupdict()['minutes'])

    # Case: '1 hora', '2 horas'
    if re.search(pattern_hours, text_small):
        for item in re.finditer(pattern_hours, text_small):
            pass
        if item.end() > end_position:
            end_position = item.end()
            hours = int(item.groupdict()['hours'])
            minutes = 0

    # Case: 'una hora'
    if re.search('una hora', text_small):
        for item in re.finditer('una hora', text_small):
            pass
        if item.end() > end_position:
            end_position = item.end()
            hours = 1
            minutes = 0

    # Case: 'un minuto'
    if re.search('un minuto', text_small):
        for item in re.finditer('una hora', text_small):
            pass
        if item.end() > end_position:
            end_position = item.end()
            hours = 0
            minutes = 1

    # Case: duration not found
    if end_position == float('-inf'):
        print("Duration hasn't been found.")
        return None

    duration = timedelta(hours=hours, minutes=minutes)
    return duration


def pipePerception(txt, pipe, end_slice=800):
    """
    Predict perception for txt between 2-4 values.
    This function only uses the first 800 characters of txt.
    txt should be an Spanish text.

    Parameters
    ----------
    txt : str
        Spanish text.

    Returns
    -------
    str
        Posible values: '☹️ 2 Mal', '🙂 3 Bien', '😀4 Muy bien'
    """

    result = pipe(txt[:end_slice])
    return result[0]['label'], result[0]['score']


# Functions and variables for tags predictions (Sentence similarity)
def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""

    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """

    post_tags = nltk.pos_tag(nltk.word_tokenize(doc))

    synsets = [wn.synsets(token, convert_tag(post_tag))[0] for (
        token, post_tag) in post_tags if (wn.synsets(token, convert_tag(post_tag)) != [])]

    return synsets


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    path_similarities = []
    for synset1 in s1:
        best_path_similarity = float('-inf')
        for synset2 in s2:
            a_path_similarity = synset1.path_similarity(synset2)
            if isinstance(a_path_similarity, float):
                if a_path_similarity > best_path_similarity:
                    best_path_similarity = a_path_similarity
        if best_path_similarity != float('-inf'):
            path_similarities.append(best_path_similarity)
    if len(path_similarities) > 0:
        return sum(path_similarities)/len(path_similarities)
    else:
        return 0


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


def is_tag(seguimiento, phrases, cut_point, name_tag):
    # this gives us a list of sentences
    sent_text = nltk.sent_tokenize(seguimiento)

    best_score = 0
    for sentence in sent_text:
        for phrase in phrases:
            score = document_path_similarity(phrase, sentence)
            if score > best_score:
                best_score = score
    log_tag = f'Best score tag {name_tag}: {best_score:.2f}. '
    print(log_tag)
    if best_score >= cut_point:
        return True, log_tag
    else:
        return False, log_tag


# GENERAL FUNCTION
def pocAudioTalent(txt, translation_tokenizer, translation_model, translation_end_slice, tagger,
                   person_end_slice, sentiment_pipe, sentiment_end_slice,
                   tag_plan_carrera_phrases=plan_carrera_phrases, tag_plan_de_carrera_cut=plan_de_carrera_cut,
                   tag_condiciones_phrases=condiciones_phrases, tag_condiciones_cut=condiciones_cut):

    # Inicialización de variables
    worker = None
    txt_en = ''
    dateTime = None
    dateTime_txt = ''
    date_txt = ''
    duration_txt = None
    perception = None
    ac_plan_de_carrera = False
    ac_condiciones = False
    log = 'LOG: '

    # NER PER extraction
    if len(txt) > person_end_slice:
        worker = personExtractionFlair(txt[:person_end_slice], tagger)
    else:
        worker = personExtractionFlair(txt, tagger)

    if worker == None:
        log += 'No se ha podido identificar el trabajador. '

    # dateTime extraction
    txt_en = translate_es_en(txt, translation_tokenizer, translation_model)
    dateTime = dateTimeExtraction(txt_en[:translation_end_slice])

    if dateTime == None:
        dateTime = datetime.now()
        dateTime_txt = dateTime.strftime('%d/%m/%Y %H:%M')
        date_txt = dateTime.strftime('%Y-%m-%d')
        log += 'No se ha podido extraer la fecha/hora del seguimiento. '
    else:
        try:
            dateTime_txt = dateTime.strftime('%d/%m/%Y %H:%M')
            date_txt = dateTime.strftime('%Y-%m-%d')
        except AttributeError:
            dateTime_txt = None

    # Duration extraction in hours
    duration = durationExtraction(txt, duration_nLastChars)
    if duration != None:
        hours = round(duration.total_seconds()/3600, 2)
        duration_txt = str(hours)
    else:
        log += 'No se ha podido extraer el tiempo requerido para el seguimiento. '
        duration_txt = 'None'

    # Perception prediction
    perception = pipePerception(txt, sentiment_pipe, sentiment_end_slice)

    # Tag ac_plan_de_carrera
    ac_plan_de_carrera, log_ac_plan_de_carrera = is_tag(
        txt_en, tag_plan_carrera_phrases, tag_plan_de_carrera_cut, 'plan de carrera')

    # Tag ac_plan_de_carrera
    ac_condiciones, log_ac_condiciones = is_tag(
        txt_en, tag_condiciones_phrases, tag_condiciones_cut, 'condiciones')

    log += log_ac_plan_de_carrera + log_ac_condiciones

    return worker, dateTime_txt, date_txt, duration_txt, perception, ac_plan_de_carrera, ac_condiciones, log


# TELEGRAM SEND MESSAGE FUNCTION
def telegram_bot_sendtext(bot_message, bot_chatID, bot_token=BOT_TOKEN):
    send_text = 'https://api.telegram.org/bot' + str(bot_token) + '/sendMessage?chat_id=' + str(
        bot_chatID) + '&parse_mode=Markdown&text=' + str(bot_message)
    response = requests.get(send_text)

    return response.json()


# DECORATOR TO RESTRICT USERS
def restricted(func):
	"""Decorator to restrict usage of func to allowed users only and replies if necessary"""
	@wraps(func)
	def wrapped(update, context, *args, **kwargs):
		user_id = update.effective_user.id
		name = update.effective_user.name

		# Access denied for users not listed in AUTHORS_ID variable
		if (user_id not in list((AUTHORS_IDS.keys()))):
			print(
			    f"Acceso no autorizado al usuario: {name}, id: {user_id}, status: {user_status}")
			update.message.reply_text(
			    f'ERROR: no estás autorizado para utilizar este bot. Tu id de Telegram es: {user_id}.')
			return  # quit function

		# Get status of the user from GO
		try:
			request_oneUser = f'https://go.opensistemas.com/users/{AUTHORS_IDS[user_id][0]}.xml?key={GO_KEY}'
			request = Request(request_oneUser)
			response_users_body = urlopen(request).read()
			dict_user = xmltodict.parse(response_users_body)['user']
			user_status = dict_user['status']
		except:
			user_status = 'unknown'

		print(f'Telegram id: {user_id}')
		print(f'GO id: {AUTHORS_IDS[user_id][0]}')
		print(f'User status: {user_status}')

		# Access denied for users that the status could not be reached
		if user_status == 'unknown':
			print(
			    f"Acceso no autorizado al usuario: {name}, id: {user_id}, status: {user_status}")
			update.message.reply_text(
			    f'ERROR: no se ha podido verificar si eres un usuario válido en GO. Te aconsejo reintentar el envío más adelante. Tu id de Telegram es: {user_id}.')
			return  # quit function

		# Access denied for users with status different than active in GO
		if user_status != '1':
			print(
			    f"Acceso no autorizado al usuario: {name}, id: {user_id}, status: {user_status}")
			update.message.reply_text(
			    f'ERROR: tu usuario está desactivado en GO. Tu id de Telegram es: {user_id}.')
			return  # quit function

		return func(update, context, *args, **kwargs)
	return wrapped


# GO API FUNCTIONS

def request_all_users(request_base, key, n_users_limit, n_users_offset=0):
	"""Devuelve un dataframe con todos los usuarios en Go. Si no se consigue conectar con la API develve un dataframe vacío"""

	def request_users(url):
		"""Función auxiliar para cada petición de usuarios"""
		try:
			request = Request(url)
			response_users_body = urlopen(request).read()
		except:
			return 0, pd.DataFrame()

		dict_users = xmltodict.parse(response_users_body)
		df_users = pd.DataFrame.from_dict(dict_users['users']['user'])
		n_users = int(dict_users['users']['@total_count'])

		return n_users, df_users

	request_url = f'{request_base}?key={key}&limit={n_users_limit}&offset={n_users_offset}'

	n_users, df_users = request_users(request_url)
	print(f'El número total de usuarios en Go es: {n_users}')

	# Calcular el número de llamadas para obtener todos los usuarios
	if n_users > 0:
		n_calls = n_users // n_users_limit
		if (n_users / n_users_limit) != n_calls:
			n_calls += 1
	else:
		n_calls = 0

	# Iteramos para traer los usuarios que falten
	for _ in range(1, n_calls):
		n_users_offset += n_users_limit

		request_url = f'{request_base}?key={key}&limit={n_users_limit}&offset={n_users_offset}'

		_, df_more_users = request_users(request_url)
		df_users = pd.concat([df_users, df_more_users], ignore_index=True)

	return df_users


def find_user_id(worker_name):
	"""Devuelve el índice del usuario en df_users cuyos nombres y apellidos tiene mayor similutud con worker_name
	   En caso de que el df_users esté vacío o el nombre de usuario sea None, devuelve None"""

	if worker_name == None:
		return None, ''

	df_users = request_all_users(request_base='https://go.opensistemas.com/users.xml',
                              key=GO_KEY,
                              n_users_limit=100)

	if len(df_users) == 0:
		return None, 'No se podido obtener el listado de usuarios de Go. '

	def make_type_consistent(s1, s2):
		"""If both objects aren't either both string or unicode instances force them to unicode"""
		if isinstance(s1, str) and isinstance(s2, str):
			return s1, s2

		elif isinstance(s1, unicode) and isinstance(s2, unicode):
			return s1, s2

		else:
			return unicode(s1), unicode(s2)

	def difflibRatio(s1, s2):
		"""Ratio implementation from FuzzyWuzzy"""

		if s1 is None:
			raise TypeError("s1 is None")
		if s2 is None:
			raise TypeError("s2 is None")
		s1, s2 = make_type_consistent(s1, s2)

		if len(s1) == 0 or len(s2) == 0:
			return 0

		m = SequenceMatcher(None, s1, s2)
		return 100 * m.ratio()

	# Precindir de los usuarios admin
	df = df_users[~df_users.mail.str.contains('admin')].copy()

	ratios = df.apply(lambda x: difflibRatio(
	    worker_name, x.firstname + ' ' + x.lastname), axis=1)
	return df_users.iloc[ratios.idxmax(), :].id, ''


def create_issue(user_go_id, project_id, description, comments, date_txt, dateTime_txt, worker, worker_id, perception, duration_txt):
	log_error = ''

	# Crear issue y obtener issue_id
	xml_create_issue = f"""
	<issue>
		<project_id>{project_id}</project_id>
		<tracker_id>{TRACKER_ID}</tracker_id>
		<status_id>{STATUS_ID}</status_id>
		<priority_id>{PRIORITY_ID}</priority_id>
		<author_id>{user_go_id}</author_id>
		<assigned_to_id>{user_go_id}</assigned_to_id>
		<subject>{'Seguimiento realizado a ' + str(worker)}</subject>
		<description>{description}</description>
		<start_date>{date_txt}</start_date>
		<due_date>{date_txt}</due_date>
		<activity_id>{ACTIVITY_ID}</activity_id>
		<custom_fields type='array'>
		<custom_field id='679' name="Date and Time Minutes">
			<value>{dateTime_txt}</value>
		</custom_field>
		<custom_field id='929' name="Worker name">
			<value>{worker_id}</value>
		</custom_field>
		<custom_field id='928' name="Perception">
			<value>{perception}</value>
		</custom_field>
	</custom_fields>
	</issue>"""

	xml_create_issue = xml_create_issue.encode('utf-8', 'xmlcharrefreplace')

	request = Request(
	    f'https://go.opensistemas.com/issues.xml?key={GO_KEY}', data=xml_create_issue, headers=headers)

	try:
		response_issue_body = urlopen(request).read()
	except HTTPError as e:
		log_error = f'No se ha podido crear el issue en GO. HTTPError code: {e.code}. '
		print(log_error)
		return 0, log_error
	except URLError as e:
		log_error = f'No se ha podido crear el issue en GO. URLError reason: {e.reason}. '
		print(log_error)
		return 0, log_error
	except:
		log_error = f'No se ha podido crear el issue en GO. '
		print(log_error)
		return 0, log_error
	else:
		print(f'RESPONSE ISSUE:\n{response_issue_body.decode()}')
		xml_issue = xmltodict.parse(response_issue_body)

		issue_id = xml_issue['issue']['id']
		print(f'El issue_id es {issue_id}')

	##  Introducir time_spent en issue previamente creado  ##
	if duration_txt != 'None':
		xml_time_entries = f"""
		<time_entry>
			<project_id>{project_id}</project_id>
			<user_id>{user_go_id}</user_id>
			<issue_id>{issue_id}</issue_id>
			<activity_id>{ACTIVITY_ID}</activity_id>
			<hours>{duration_txt}</hours>
		</time_entry>"""

		xml_time_entries = xml_time_entries.encode('utf-8', 'xmlcharrefreplace')

		request = Request(
		    f'https://go.opensistemas.com/time_entries.xml?key={GO_KEY}', data=xml_time_entries, headers=headers)

		try:
			response_time_body = urlopen(request).read()
		except HTTPError as e:
			log_error = f'No se ha podido crear el time spent en GO. HTTPError code: {e.code}. '
			print(log_error)
		except URLError as e:
			log_error = f'No se ha podido crear el time spent en GO. URLError reason: {e.reason}. '
			print(log_error)
		except:
			log_error = f'No se ha podido crear el time spent en GO. '
			print(log_error)
		else:
			print(F'RESPONSE TIME ENTRY:\n{response_time_body.decode()}')

	##  Cear comentario con info de TAGS, LOSGs y archivo de audio  ##
	xml_issue_notes = f"<issue><notes>{comments}</notes></issue>"
	xml_issue_notes = xml_issue_notes.encode('utf-8')

	request = Request(
	    f'https://go.opensistemas.com/issues/{issue_id}.xml?key={GO_KEY}', data=xml_issue_notes, headers=headers, method="PUT")

	try:
		response_issue_notes = urlopen(request).read()
	except HTTPError as e:
		log_error = f'No se ha podido crear el comentario en GO. HTTPError code: {e.code}. '
		print(log_error)
	except URLError as e:
		log_error = f'No se ha podido crear el comentario en GO. URLError reason: {e.reason}. '
		print(log_error)
	except:
		log_error = f'No se ha podido crear el comentario en GO. '
		print(log_error)
	else:
		print(f'RESPONSE COMMENT:\n{response_issue_notes.decode()}')

	return issue_id, log_error
