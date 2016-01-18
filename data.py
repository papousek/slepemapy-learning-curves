import pandas
import proso.analysis as pa
from proso.geography.dfutil import iterdicts
import numpy
import os
from geoip import geolite2


SETUP = {
    6: 'R-A',
    7: 'R-R',
    8: 'A-A',
    9: 'A-R',
    14: 50,
    15: 35,
    16: 20,
    17: 5,
}


def load_data(answer_limit, filter_invalid_tests=True, filter_invalid_response_time=True, rolling_success=False):
    answers = pandas.read_csv('./answers.csv', index_col=False, parse_dates=['time'])
    flashcards = pandas.read_csv('./flashcards.csv', index_col=False)
    user_ip = pandas.read_csv('./ip_address.csv', index_col=False)
    difficulty = pandas.read_csv('./difficulty.csv')

    answers['experiment_setup_name'] = answers['experiment_setup_id'].apply(lambda i: SETUP[i])

    valid_users = map(lambda x: x[0], filter(lambda x: x[1] >= answer_limit, answers.groupby('user_id').apply(len).to_dict().items()))
    answers = answers[answers['user_id'].isin(valid_users)]

    if filter_invalid_response_time:
        invalid_users = answers[answers['response_time'] < 0]['user_id'].unique()
        answers = answers[~answers['user_id'].isin(invalid_users)]

    answers = pandas.merge(answers, flashcards, on='item_id', how='inner')
    answers = pandas.merge(answers, difficulty, on='item_id', how='inner')

    if filter_invalid_tests:
        invalid_users = answers[answers['context_id'] == 17]['user_id'].unique()
        answers = answers[~answers['user_id'].isin(invalid_users)]

        invalid_users = set()
        last_user = None
        last_context = None
        counter = None
        for row in iterdicts(answers.sort(['user_id', 'context_name', 'term_type', 'id'])):
            if last_user != row['user_id'] or last_context != (row['context_name'], row['term_type']):
                last_user = row['user_id']
                last_context = (row['context_name'], row['term_type'])
                counter = 0
            if row['metainfo_id'] == 1 and counter % 10 != 0:
                invalid_users.add(row['user_id'])
            counter += 1
        answers = answers[~answers['user_id'].isin(invalid_users)]

    answers = pa.decorate_session_number(answers, 3600 * 10)
    answers = decorate_school(answers, user_ip=user_ip)
    if rolling_success:
        answers = pa.decorate_last_in_session(answers)
        answers = pa.decorate_rolling_success(answers)
    return answers.sort(['user_id', 'id'])


def decorate_school(answers, threshold_in_minutes=10, threshold_people=10, user_ip=None, time_column='time', user_column='user_id'):
    user_first_answers = answers.drop_duplicates([user_column])[[user_column, time_column]].set_index(user_column)
    user_last_answers = answers.drop_duplicates([user_column], take_last=True)[[user_column, time_column]].set_index(user_column)
    user_first_last = user_first_answers.join(user_last_answers, how='right', lsuffix='_first', rsuffix='_last')

    def _get_classrooms(column):
        users = []
        total = 0
        classrooms = []
        for user_id, time_diff in user_first_last.sort(column)[column].diff().iteritems():
            time_diff = time_diff / numpy.timedelta64(1, 'm') if not isinstance(time_diff, pandas.tslib.NaTType) else 0
            if time_diff and total + time_diff <= threshold_in_minutes:
                users.append(user_id)
                total += time_diff
            else:
                if len(users) > threshold_people:
                    classrooms.append(set(users))
                users = []
                total = 0
        return classrooms

    classrooms_first = _get_classrooms(time_column + '_first')
    classrooms_last = _get_classrooms(time_column + '_last')
    classrooms = []
    for c_first in classrooms_first:
        for c_last in classrooms_last:
            c_intersection = c_first & c_last
            if len(c_intersection) > threshold_people:
                classrooms.append(c_intersection)
    classroom_users = [u for c in classrooms for u in c]

    if user_ip is not None:
        school_ips = map(lambda xs: xs[0], filter(lambda (i, n): n >= 10, user_ip.groupby('ip_address').apply(lambda us: len(us[user_column].unique())).to_dict().items()))
        school_users = user_ip[user_ip['ip_address'].isin(school_ips)][user_column].unique()
    else:
        school_users = None
    school_detected = answers[user_column].apply(lambda u: u in classroom_users or (school_users is not None and u in school_users))
    if 'school' in answers:
        answers['school'] = answers['school'] | school_detected
    else:
        answers['school'] = school_detected
    return answers


def get_country(ip):
    lookup = geolite2.lookup(ip)
    return numpy.nan if lookup is None else lookup.country


def prepare_public_data(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    answers = pa.get_raw_data('answers', load_data, 'experiment_cache', answer_limit=1)[[
        'id', 'time', 'response_time', 'item_answered_id', 'item_asked_id', 'user_id',
        'guess', 'metainfo_id', 'direction', 'experiment_setup_name', 'context_name', 'session_id'
    ]]
    flashcards = pandas.read_csv('./flashcards.csv', index_col=False)
    feedback = pandas.read_csv('./ratings.csv', index_col=False)
    session_ip = pandas.read_csv('./ip_address.csv', index_col=False)[['sesion_id', 'ip_address']]

    session_ip['session_id'] = session_ip['sesion_id']
    session_ip = session_ip[['session_id', 'ip_address']]
    session_ip['ip_country'] = session_ip['ip_address'].apply(get_country)
    ips = session_ip['ip_address'].unique()
    ip_ids = dict(zip(ips, range(1, len(ips) + 1)))
    ip_ids[numpy.nan] = numpy.nan
    session_ip['ip_id'] = session_ip['ip_address'].apply(lambda i: ip_ids[i])
    session_ip = session_ip[['session_id', 'ip_country', 'ip_id']]
    answers = pandas.merge(answers, session_ip, on='session_id', how='inner')

    answers['options'] = answers['guess'].apply(lambda g: 0 if g == 0 else int(round(1 / g)))
    answers['reference'] = answers['metainfo_id'] == 1
    answers['condition'] = answers['experiment_setup_name']
    answers = pandas.merge(answers, flashcards[['item_id', 'term_name']], left_on='item_asked_id', right_on='item_id', how='inner')
    answers['term_asked_name'] = answers['term_name']
    del answers['term_name']
    answers = pandas.merge(answers, flashcards[['item_id', 'term_name']], left_on='item_answered_id', right_on='item_id', how='inner')
    answers['term_answered_name'] = answers['term_name']
    del answers['term_name']

    term_type_trans = {
        'region_cz': 'region',
        'bundesland': 'region',
        'region_it': 'region',
        'autonomous_Comunity': 'region',
        'province': 'region'
    }

    term_type_dict = flashcards.set_index('item_id')['term_type'].to_dict()
    us_state_items = flashcards[(flashcards['context_name'] == 'United States') & (flashcards['term_type'] == 'state')]['item_id'].unique()
    flashcards['term_type'] = flashcards['item_id'].apply(
        lambda i: term_type_trans.get(term_type_dict[i], term_type_dict[i]) if i not in us_state_items else 'region'
    )
    answers = pandas.merge(answers, flashcards[['item_id', 'term_type']], left_on='item_asked_id', right_on='item_id', how='inner')

    answers = answers[[
        'id', 'time', 'response_time', 'item_answered_id', 'item_asked_id', 'user_id',
        'options', 'reference', 'direction', 'condition', 'context_name',
        'term_type', 'term_asked_name', 'term_answered_name', 'ip_country', 'ip_id'
        ]]
    answers.to_csv('{}/answers.csv'.format(dest), index=False)

    feedback = feedback[feedback['user_id'].isin(answers['user_id'].unique())]
    feedback_values = {1: 'easy', 2: 'appropriate', 3: 'difficult'}
    feedback['value'] = feedback['value'].apply(lambda f: feedback_values[f])
    feedback.to_csv('{}/feedback.csv'.format(dest), index=False)


if __name__ == "__main__":
    prepare_public_data('public')
