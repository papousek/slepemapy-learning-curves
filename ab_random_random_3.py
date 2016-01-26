# works with:
#   http://data-private.slepemapy.cz/ab-experiment-random-parts-3.zip

import data as datalib
import proso.analysis as pa
import pandas
import numpy
from collections import defaultdict
from proso.metric import binomial_confidence_mean
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from scipy.optimize import curve_fit
from scipy.stats import weibull_min
import os
import scikits.bootstrap as bootstrap
from proso.geography.dfutil import iterdicts
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import math
import random
from matplotlib.gridspec import GridSpec


SNS_STYLE = {
    'style': 'white',
    'font_scale': 1.8,
}
MARKER_SIZE = 10
#HATCHES=['//', '', '\\\\', '/', '\\', '.', '*', 'O', '-', '+', 'x', 'o']
HATCHES=[''] * 10
LINES=['-', '--', '-.', ':']
sns.set(**SNS_STYLE)

MARKERS = "dos^"
COLORS = sns.color_palette('gray')[2:6][::-1]
#TARGET_DIR = 'europe-state'
TARGET_DIR = '.'


def fit_learning_curve(data, length=10, user_length=None, context_answer_limit=100, reverse=False, bootstrap_samples=100):
    confidence_vals = [[] for i in range(length)]

    def _fit_learning_curve(series):
        references_by_attempt = map(lambda references: [r for r in references if r is not None], zip(*series))
        learning_curve = map(lambda xs: (numpy.mean(xs), len(xs)), references_by_attempt)

        def _learn_fun(attempt, a, k):
            return a * (1.0 / (attempt + 1) ** k)

        opt, _ = curve_fit(
            _learn_fun,
            numpy.arange(len(learning_curve)),
            numpy.array(map(lambda x: x[0], learning_curve)),
            sigma=numpy.array(map(lambda x: numpy.sqrt((x[0] * (1 - x[0])) / x[1]), learning_curve))
        )
        fit = map(lambda attempt: _learn_fun(attempt, opt[0], opt[1]), range(length))
        for i, r in enumerate(fit):
            confidence_vals[i].append(r)
        return fit[-1]

    full_series = reference_series(data, length=length, user_length=user_length,
        context_answer_limit=context_answer_limit, reverse=reverse)
    if len(full_series) == 0:
        return [{
            'value': None,
            'confidence_interval_min': None,
            'confidence_interval_max': None,
        }] * length
    series = list(zip(*full_series)[1])
    bootstrap.ci(series, _fit_learning_curve, method='pi', n_samples=bootstrap_samples)

    def _aggr(rs):
        return {
            'value': numpy.median(rs),
            'confidence_interval_min': numpy.percentile(rs, 2.5),
            'confidence_interval_max': numpy.percentile(rs, 97.5),
        }
    return map(_aggr, confidence_vals)


def answers_density(data, length=10):
    trans_fun = lambda x: (x - 1) / 10 + 0.5
    nums = numpy.array(map(trans_fun,
        data.groupby('user_id').apply(len).to_dict().values()
    ))
    available_values = sorted(list(set([trans_fun(i) for i in range(1, length + 1)])))
    return [len([num for num in nums if num == i]) / float(len(nums)) for i in available_values]


def fit_weibull(data, length=10):
    trans_fun = lambda x: (x - 1) / 10 + 0.5
    x = numpy.array(sorted(list(set([trans_fun(i) for i in numpy.arange(1, length + 1)]))))
    confidence_vals = [[] for i in range(len(x))]
    nums = numpy.array(data.groupby('user_id').apply(len).to_dict().values())
    nums_groupped = defaultdict(list)
    for num in nums:
        nums_groupped[(num - 1) / 10].append(num)
    nums_avg = {key: numpy.mean(values) / 10.0 for (key, values) in nums_groupped.iteritems()}
    nums_trans = [nums_avg[(num - 1) / 10] for num in nums]

    fit_values = weibull_min.fit(nums_trans, floc=0)
    fit = weibull_min.pdf(x, *fit_values)
    for i, f in enumerate(fit):
        confidence_vals[i] = f

    def _aggr(r):
        return {
            'value': r,
            'confidence_interval_min': r,
            'confidence_interval_max': r,
        }
    return {
        'serie': map(_aggr, confidence_vals),
        'params': list(fit_values),
    }


def _histogram(xs, bins=10):
    hist, bins = numpy.histogram(xs, bins=bins)
    return {
        'hist': list(hist),
        'bins': list(bins),
    }


def _sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def _resample(xs):
    xs = numpy.array(xs)
    upper_bound = len(xs) - 1
    return [xs[random.randint(0, upper_bound)] for _ in xrange(xs)]


def _format_group_name(group_name):
    try:
        group_name = int(group_name)
        return 'C{}'.format(group_name)
    except:
        return group_name


def _format_context(context_name):
    return context_name.replace('Czech Rep.', 'CZ').replace('United States', 'US').replace('region_cz', 'region')


def _format_label(label, size=15):
    try:
        label = unicode(label)
    except:
        pass
    if len(label) <= size:
        return label
    split_label = label.split(' ')
    if len(split_label) == 1:
        return split_label[:size]
    return (' '.join([l[0] for l in split_label[:-1]]) + ' ' + split_label[-1])[:size]


def _format_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return '{}s'.format(seconds)
    if seconds < 3600:
        if seconds % 60 > 0:
            return '{}m, {}'.format(seconds / 60, _format_time(seconds % 60))
        else:
            return '{}m'.format(seconds / 60)
    if seconds < 3600 * 24:
        if seconds % 3600 > 0:
            return '{}h, {}'.format(seconds / 3600, _format_time(seconds % 3600))
        else:
            return '{}h'.format(seconds / 3600)
    else:
        if seconds % (3600 * 24) > 0:
            return '{}d, {}'.format(seconds / (3600 * 24), _format_time(seconds % (3600 * 24)))
        else:
            return '{}d'.format(seconds / (3600 * 24))


def _savefig(basename, filename):
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    plt.tight_layout()
    filename = '{}/{}_{}.png'.format(TARGET_DIR, basename, filename)
    plt.savefig(filename)
    print filename


def milestone_progress(data, length, with_confidence=False):
    user_answers = data.groupby('user_id').apply(len).to_dict().values()

    def _progress_confidence(i):
        xs = map(lambda x: x > i, filter(lambda x: x > i - 10, user_answers))
        value, confidence = binomial_confidence_mean(xs)
        return {
            'value': value,
            'confidence_interval_min': confidence[0],
            'confidence_interval_max': confidence[1],
        }

    result = []
    for i in range(10, 10 * length + 1, 10):
        result.append(_progress_confidence(i))
    return result


def practice_error(data):
    data = data[data['metainfo_id'] != 1]
    value, confidence = binomial_confidence_mean(data['item_asked_id'] != data['item_answered_id'])
    return {
        'value': value,
        'confidence_interval_min': confidence[0],
        'confidence_interval_max': confidence[1],
        'size': len(data),
    }


def returning(data):
    returning = data.groupby('user_id').apply(lambda g: g['session_number'].max() > 0).values
    value, confidence = binomial_confidence_mean(returning)
    return {
        'value': value,
        'confidence_interval_min': confidence[0],
        'confidence_interval_max': confidence[1],
        'size': len(returning),
    }


def output_rolling_success(data):
    total = float(len(data))
    data = data[numpy.isfinite(data['rolling_success'])]
    return data.groupby('rolling_success').apply(lambda x: len(x) / total).to_dict()


def stay_on_rolling_success(data):
    data = data[numpy.isfinite(data['rolling_success'])]

    def _stay(group):
        value, confidence = binomial_confidence_mean(group['stay'])
        return {
            'value': value,
            'confidence_interval_min': confidence[0],
            'confidence_interval_max': confidence[1],
        }

    return (data.
        groupby(['user_id', 'rolling_success']).
        apply(lambda x: sum(~x['last_in_session']) / float(len(x))).
        reset_index().
        rename(columns={0: 'stay'}).
        groupby('rolling_success').
        apply(_stay).
        to_dict())


def attrition_bias(data, length=6, context_answer_limit=100):

    def _attrition_bias(group):
        if len(group) < context_answer_limit:
            return []
        user_answers_dict = defaultdict(list)
        for row in iterdicts(group):
            user_answers_dict[row['user_id']].append(row['item_asked_id'] != row['item_answered_id'])
        return user_answers_dict.values()

    user_answers = [answers for context_answers in data.groupby(['context_name', 'term_type']).apply(_attrition_bias) for answers in context_answers]
    result = []
    for i in range(length):
        value, confidence = binomial_confidence_mean([answers[0] for answers in user_answers if len(answers) > i])
        result.append({
            'value': value,
            'confidence_interval_min': confidence[0],
            'confidence_interval_max': confidence[1],
        })
    return result


def item_occurences(group):

    def _occurs(_group):
        last_user = None
        i = None
        result = defaultdict(lambda: [[], 0, None])
        for user_id, item_asked_id, difficulty, term_name in _group.drop_duplicates(['user_id', 'item_asked_id'])[['user_id', 'item_asked_id', 'difficulty', 'term_name']].values:
            if last_user != user_id:
                last_user = user_id
                i = 0
            else:
                result[item_asked_id][0].append(i)
                result[item_asked_id][1] = difficulty
                result[item_asked_id][2] = term_name
            i += 1
        return result
    int_result = group.groupby(['context_name', 'term_type']).apply(_occurs).to_dict()
    return {i: {
        'value': numpy.median(os[0]),
        'difficulty': os[1],
        'term_name': os[2],
    } for data in int_result.itervalues() for (i, os) in data.iteritems()}


def item_frequencies(group):
    def _freq(_group):
        total = float(len(_group))
        return _group.groupby('item_asked_id').apply(lambda xs: (len(xs) / total, xs['difficulty'].values[0], xs['term_name'].values[0])).to_dict()
    int_result = group.groupby(['context_name', 'term_type']).apply(_freq).to_dict()
    return {i: {
        'value': f[0],
        'difficulty': f[1],
        'term_name': f[2]
    } for data in int_result.itervalues() for (i, f) in data.iteritems()}


def progress(data, length=60):
    user_answers = data.groupby('user_id').apply(len).to_dict().values()

    def _progress_confidence(i):
        xs = map(lambda x: x > i, user_answers)
        value, confidence = binomial_confidence_mean(xs)
        return {
            'value': value,
            'confidence_interval_min': confidence[0],
            'confidence_interval_max': confidence[1],
        }

    result = []
    for i in range(length):
        result.append(_progress_confidence(i))
    return result


def reference_series(data, length=10, context_answer_limit=100, reverse=False, user_length=None, save_fun=None, require_length=True):

    if save_fun is None:
        save_fun = lambda row: row['item_asked_id'] != row['item_answered_id']

    def _context_series(group):
        if len(group) < context_answer_limit:
            return []
        user_answers_dict = defaultdict(list)
        for row in iterdicts(group):
            user_answers_dict[row['user_id']].append(save_fun(row))

        def _user_answers(user, answers):
            if reverse:
                answers = answers[::-1]
            if require_length:
                answers = answers[:min(len(answers), length)]
                nones = [None for _ in range(length - len(answers))]
            else:
                nones = []
            if reverse:
                answers = answers[::-1]
                return user, nones + answers
            else:
                return user, answers + nones

        return [
            _user_answers(user, answers)
            for user, answers in user_answers_dict.iteritems()
            if user_length is None or len(answers) >= user_length
        ]
    return [val for (_, vals) in data.groupby(['context_name', 'term_type']).apply(_context_series).iteritems() for val in vals]


def learning_slope(data, data_all, length=10, user_length=None, context_answer_limit=100, bootstrap_samples=100):
    if len(data) < 1000:
        return {
            'value': None,
            'confidence_interval_min': None,
            'confidence_interval_max': None,
        }

    data_users = set(data['user_id'].unique())
    series_all = reference_series(data_all, length=length, context_answer_limit=context_answer_limit)

    def _fit_learning_curve(series):
        mean_all_first = numpy.mean([serie[1][0] for serie in series])
        series = zip(*filter(lambda (u, s): u in data_users, series))[1]
        references_by_attempt = map(lambda references: [r for r in references if r is not None], zip(*series))
        learning_curve = map(lambda xs: (numpy.mean(xs), len(xs)), references_by_attempt)

        def _learn_fun(attempt, k):
            return mean_all_first / (attempt + 1) ** k

        opt, _ = curve_fit(
            _learn_fun,
            numpy.arange(len(learning_curve)),
            numpy.array(map(lambda x: x[0], learning_curve)),
            sigma=numpy.array(map(lambda x: 1.0 / numpy.sqrt(x[1] + 1), learning_curve))
        )
        return opt[0]

    confidence = bootstrap.ci(series_all, _fit_learning_curve, method='pi', n_samples=bootstrap_samples)

    return {
        'value': _fit_learning_curve(series_all),
        'confidence_interval_min': confidence[0],
        'confidence_interval_max': confidence[1],
    }


def learning_gain(data, length=10, user_length=None, context_answer_limit=100, bootstrap_samples=100):
    if len(data) < 100:
        return {
            'value': None,
            'confidence_interval_min': None,
            'confidence_interval_max': None,
        }
    if user_length is None:
        user_length = 2
    half = max(1, user_length / 2)
    series = zip(*reference_series(data, length=length, user_length=user_length,
        context_answer_limit=context_answer_limit, require_length=False))[1]
    pre_post = map(lambda serie: (serie[:half], serie[half:]), series)

    def _gain(data):
        pre, post = zip(*data)
        pre_mean = numpy.mean([x for xs in pre for x in xs])
        post_mean = numpy.mean([x for xs in post for x in xs])
        return (pre_mean - post_mean) / pre_mean

    confidence = bootstrap.ci(pre_post, _gain, method='pi', n_samples=bootstrap_samples)

    return {
        'value': _gain(pre_post),
        'confidence_interval_min': confidence[0],
        'confidence_interval_max': confidence[1],
    }


def learning_curve(data, length=10, user_length=None, context_answer_limit=100, reverse=False):
    full_series = reference_series(data, length=length, user_length=user_length,
        context_answer_limit=context_answer_limit, reverse=reverse)
    if len(full_series) == 0:
        series = [[]] * length
    else:
        series = zip(*full_series)[1]
    references_by_attempt = map(lambda references: [r for r in references if r is not None], zip(*series))

    def _weighted_mean(xs):
        value, confidence = binomial_confidence_mean(xs)
        return {
            'value': value,
            'confidence_interval_min': confidence[0],
            'confidence_interval_max': confidence[1],
            'size': len(xs),
        }
    return map(_weighted_mean, references_by_attempt)


def test_questions(data, length=100):
    last_user = None
    last_context = None
    counter = None
    result = defaultdict(lambda: 0)
    for row in iterdicts(data.sort(['user_id', 'context_name', 'term_type', 'id'])):
        if last_user != row['user_id'] or last_context != (row['context_name'], row['term_type']):
            last_user = row['user_id']
            last_context = (row['context_name'], row['term_type'])
            counter = 0
        if row['metainfo_id'] == 1 and counter < length:
            result[counter] += 1
        counter += 1
    return dict(result.items())


def response_time_curve(data, length=10, user_length=None, context_answer_limit=100, correctness=True, reverse=False):

    series = zip(*reference_series(data, length=length, user_length=user_length,
        context_answer_limit=context_answer_limit, reverse=reverse,
        save_fun=lambda row: (row['response_time'] if correctness is None or (row['item_asked_id'] == row['item_answered_id']) == correctness else None)))[1]
    references_by_attempt = map(lambda references: [r for r in references if r is not None], zip(*series))

    def _aggregate(xs):
        xs = filter(lambda x: x is not None, xs)
        return {
            'value': numpy.median(xs),
        }
    return map(_aggregate, references_by_attempt)


def learning_points(data, length=5):
    context_answers = defaultdict(dict)
    for row in iterdicts(data):
        user_answers = context_answers[row['term_type'], row['context_name']]
        if row['user_id'] in user_answers:
            user_answers[row['user_id']].append((
                row['time'],
                (row['time'] - user_answers[row['user_id']][-1][0]).total_seconds(),
                len(user_answers[row['user_id']]),
                row['item_asked_id'] == row['item_answered_id']
            ))
        else:
            user_answers[row['user_id']] = [(
                row['time'],
                0,
                0,
                row['item_asked_id'] == row['item_answered_id']
            )]
    answers = [a for user_answers in context_answers.itervalues() for uas in user_answers.itervalues() for a in uas]
    thresholds = numpy.percentile(map(lambda xs: xs[1], answers), range(10, 100, 10))
    thresholds = [60, 120, 300, 600, 3600, 3600 * 24]
    thresholds = zip([0] + thresholds, thresholds + [60 * 60 * 24 * 375])
    result = []
    for attempt in range(length):
        for lower, upper in thresholds:
            filtered = map(lambda xs: xs[3], filter(lambda xs: xs[1] >= lower and xs[1] < upper and xs[2] == attempt, answers))
            result.append((attempt, lower, None if len(filtered) < 30 else numpy.mean(filtered), len(filtered)))
    return result


def meta(data):
    return {
        'answers': len(data),
        'users': len(data['user_id'].unique()),
    }


def compute_experiment_data(term_type=None, term_name=None, context_name=None, answer_limit=10, curve_length=5, progress_length=60, filter_invalid_tests=True, with_confidence=False, keys=None, contexts=False, filter_invalid_response_time=True, school=None, bootstrap_samples=200, density_length=100):
    compute_rolling_success = keys is None or len(set(keys) & {'output_rolling_success', 'stay_on_rolling_success'}) != 0
    data = pa.get_raw_data('answers',  datalib.load_data, 'experiment_cache', debug=True,
        answer_limit=answer_limit, filter_invalid_tests=filter_invalid_tests,
        filter_invalid_response_time=filter_invalid_response_time, rolling_success=compute_rolling_success
    )
    if term_type is not None:
        if not isinstance(term_type, list):
            term_type = [term_type]
        data = data[data['term_type'].isin(term_type)]
    if term_name is not None:
        if not isinstance(term_name, list):
            term_name = [term_name]
        data = data[data['term_name'].isin(term_name)]
    if context_name is not None:
        if not isinstance(context_name, list):
            context_name = [context_name]
        data = data[data['context_name'].isin(context_name)]
    if school is not None:
        data = data[data['school'] == school]

    def _group_experiment_data(data_all, extended=False):
        data = data_all[data_all['metainfo_id'] == 1]
        groupped = data.groupby('experiment_setup_name')
        groupped_all = data_all.groupby('experiment_setup_name')
        result = {
            'meta': groupped_all.apply(meta).to_dict(),
            'meta_all': meta(data_all),
        }
        if keys is None or 'learning_points' in keys:
            result['learning_points'] = groupped.apply(lambda g: learning_points(g, length=curve_length)).to_dict()
        if keys is None or 'error_hist' in keys:
            result['error_hist'] = groupped_all.apply(
                lambda group: _histogram(group.groupby('user_id').apply(lambda g: (g['item_asked_id'] != g['item_answered_id']).mean()).to_dict().values(), bins=numpy.arange(0, 1.1, step=0.1))
            ).to_dict()
        if keys is None or 'learning_points_all' in keys:
            result['learning_points_all'] = learning_points(data, length=curve_length)
        if keys is None or 'learning_curve_all' in keys:
            result['learning_curve_all'] = groupped.apply(lambda g: learning_curve(g, length=curve_length)).to_dict()
        if keys is None or 'learning_curve_all_reverse' in keys:
            result['learning_curve_all_reverse'] = groupped.apply(lambda g: learning_curve(g, length=curve_length, reverse=True)).to_dict()
        if keys is None or 'learning_curve' in keys:
            result['learning_curve'] = groupped.apply(lambda g: learning_curve(g, length=curve_length, user_length=curve_length)).to_dict()
        if keys is None or 'learning_curve_reverse' in keys:
            result['learning_curve_reverse'] = groupped.apply(lambda g: learning_curve(g, length=curve_length, user_length=curve_length, reverse=True)).to_dict()
        if keys is None or 'response_time_curve_correct_all' in keys:
            result['response_time_curve_correct_all'] = groupped.apply(lambda g: response_time_curve(g, length=curve_length)).to_dict()
        if keys is None or 'response_time_curve_correct' in keys:
            result['response_time_curve_correct'] = groupped.apply(lambda g: response_time_curve(g, length=curve_length, user_length=curve_length)).to_dict()
        if keys is None or 'response_time_curve_wrong_all' in keys:
            result['response_time_curve_wrong_all'] = groupped.apply(lambda g: response_time_curve(g, length=curve_length, correctness=False)).to_dict()
        if keys is None or 'response_time_curve_wrong' in keys:
            result['response_time_curve_wrong'] = groupped.apply(lambda g: response_time_curve(g, length=curve_length, user_length=curve_length, correctness=False)).to_dict()
        if keys is None or 'test_questions_hist' in keys:
            result['test_questions_hist'] = groupped_all.apply(lambda g: test_questions(g, length=progress_length)).to_dict()
        if keys is None or 'attrition_bias' in keys:
            result['attrition_bias'] = groupped.apply(lambda g: attrition_bias(g, curve_length)).to_dict()
        if keys is None or 'learning_curve_fit_all' in keys:
            result['learning_curve_fit_all'] = groupped.apply(lambda g: fit_learning_curve(g, length=curve_length, bootstrap_samples=bootstrap_samples)).to_dict()
        if keys is None or 'learning_curve_fit' in keys:
            result['learning_curve_fit'] = groupped.apply(lambda g: fit_learning_curve(g, length=curve_length, user_length=curve_length, bootstrap_samples=bootstrap_samples)).to_dict()
        if keys is None or 'item_occurences' in keys:
            result['item_occurences'] = groupped_all.apply(item_occurences).to_dict()
        if keys is None or 'item_frequencies' in keys:
            result['item_frequencies'] = groupped_all.apply(item_frequencies).to_dict()
        if keys is None or 'practice_error' in keys:
            result['practice_error'] = groupped_all.apply(practice_error).to_dict()
        if keys is None or 'learning_gain' in keys:
            result['learning_gain'] = groupped.apply(lambda g: learning_gain(g, bootstrap_samples=bootstrap_samples)).to_dict()
        if keys is None or 'learning_slope' in keys:
            result['learning_slope'] = groupped.apply(lambda g: learning_slope(g, data, bootstrap_samples=bootstrap_samples)).to_dict()

        if extended:
            if keys is None or 'response_time_curve_global' in keys:
                result['response_time_curve_global'] = groupped_all.apply(lambda g: response_time_curve(g, length=progress_length, correctness=None)).to_dict()
            if keys is None or 'learning_curve_global' in keys:
                result['learning_curve_global'] = groupped_all.apply(lambda g: learning_curve(g, length=progress_length)).to_dict()
            if keys is None or 'learning_curve_fit_reverse' in keys:
                result['learning_curve_fit_reverse'] = groupped.apply(lambda g: fit_learning_curve(g, length=curve_length, user_length=curve_length, reverse=True, bootstrap_samples=bootstrap_samples)).to_dict()
            if keys is None or 'progress' in keys:
                result['progress'] = groupped_all.apply(lambda g: progress(g, length=progress_length)).to_dict()
            if keys is None or 'progress_milestones' in keys:
                result['progress_milestones'] = groupped_all.apply(lambda g: milestone_progress(g, length=progress_length / 10, with_confidence=with_confidence)).to_dict()
            if keys is None or 'returning' in keys:
                result['returning'] = groupped_all.apply(lambda g: returning(g)).to_dict()
            if keys is None or 'stay_on_rolling_success' in keys:
                result['stay_on_rolling_success'] = groupped_all.apply(lambda g: stay_on_rolling_success(g)).to_dict()
            if keys is None or 'output_rolling_success' in keys:
                result['output_rolling_success'] = groupped_all.apply(lambda g: output_rolling_success(g)).to_dict()

            if keys is None or 'weibull' in keys:
                result['weibull'] = groupped_all.apply(lambda g: fit_weibull(g, density_length)).to_dict()
            if keys is None or 'answers_density' in keys:
                result['answers_density'] = groupped_all.apply(lambda g: answers_density(g, density_length)).to_dict()
        return result
    result = {
        'all': _group_experiment_data(data, extended=True),
    }
    if contexts:
        result['contexts'] = {
            '{}, {}'.format(context_name, term_type): value
            for ((context_name, term_type), value) in data.groupby(['context_name', 'term_type']).apply(_group_experiment_data).to_dict().iteritems()
        }
    return result


def plot_line(data, with_confidence=True, markevery=None, invert=False, setups=None, marker=True, xs=None, ax=None, skip_line=False, markersize=MARKER_SIZE, **kwargs):
    if ax is None:
        ax = plt.gca()
    if markevery is not None:
        kwargs['markevery'] = markevery
    for i, (group_name, group_data) in enumerate(sorted(data.items())):
        if xs is None:
            group_xs = numpy.arange(1, len(group_data) + 1)
        else:
            group_xs = xs
        if setups is not None and group_name not in setups:
            continue
        group_kwargs = dict(kwargs)
        if marker:
            group_kwargs['marker'] = MARKERS[i]
        ys = numpy.array(map(lambda x: x['value'], group_data))
        if invert:
            ys = 1 - ys
        linestyle = '' if skip_line else LINES[i]
        ax.plot(group_xs, ys, label=_format_group_name(group_name), color=COLORS[i], **group_kwargs)
        if with_confidence:
            plt.fill_between(
                group_xs,
                map(lambda x: x['confidence_interval_min'], group_data),
                map(lambda x: x['confidence_interval_max'], group_data),
                color=COLORS[i], alpha=0.35
            )
        ax.set_xlim(min(group_xs), max(group_xs))


def ylim_learning_curve():
    plt.ylim(0, 0.6)


def plot_experiment_data(experiment_data, filename, context_rows=3, context_cols=3):
    if {'learning_curve_all', 'learning_curve', 'learning_curve_reverse', 'learning_curve_fit_all', 'learning_curve_fit', 'learning_curve_fit_reverse'} <= set(experiment_data.get('all', {}).keys()):
        rcParams['figure.figsize'] = 22.5, 10
        plt.subplot(231)
        ylim_learning_curve()
        plt.title('All learners')
        plot_line(experiment_data['all']['learning_curve_all'])
        plt.ylabel('Error rate')
        plt.legend(loc=1, frameon=True, ncol=2)

        plt.subplot(232)
        plot_line(experiment_data['all']['learning_curve'])
        ylim_learning_curve()
        plt.title('Filtered learners')

        plt.subplot(233)
        plot_line(experiment_data['all']['learning_curve_reverse'])
        ylim_learning_curve()
        plt.title('Filtered learners, reverse')

        plt.subplot(234)
        ylim_learning_curve()
        plot_line(experiment_data['all']['learning_curve_fit_all'], with_confidence=False)
        plt.ylabel('Error rate')
        plt.xlabel('Attempt')

        plt.subplot(235)
        plot_line(experiment_data['all']['learning_curve_fit'], with_confidence=False)
        ylim_learning_curve()
        plt.xlabel('Attempt')

        plt.subplot(236)
        plot_line(experiment_data['all']['learning_curve_fit_reverse'], with_confidence=False)
        ylim_learning_curve()
        plt.xlabel('Attempt')

        _savefig(filename, 'learning_curve_combined')
        plt.close()

        rcParams['figure.figsize'] = 15, 5
        plt.subplot(121)
        ylim_learning_curve()
        plt.title('Raw data')
        plot_line(experiment_data['all']['learning_curve_all'])
        plt.ylabel('Error rate')
        plt.xlabel('Attempt')
        plt.legend(loc=1, frameon=True, ncol=2)
        plt.subplot(122)
        ylim_learning_curve()
        plt.title('Learning curve')
        plot_line(experiment_data['all']['learning_curve_fit_all'])
        plt.xlabel('Attempt')
        _savefig(filename, 'learning_curve_combined_simple')
        plt.close()


    if 'progress' in experiment_data.get('all', {}) and 'weibull' in experiment_data.get('all', {}) and 'answers_density' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 15, 5
        plt.subplot(121)
        for i, (group_name, data) in enumerate(sorted(experiment_data['all']['progress'].items())):
            plt.plot(numpy.arange(len(data)) + 1, map(lambda x: x['value'], data), label=_format_group_name(group_name), color=COLORS[i])

        plt.legend(loc=1, frameon=True)
        plt.xlabel('Number of attempts')
        plt.ylabel('Proportion of learners')

        plt.subplot(122)
        ys = map(lambda x: x['value'], experiment_data['all']['weibull'][50]['serie'])
        plt.plot(numpy.arange(len(ys)) + 1, ys, '--', color='red', label='Weibull distribution')
        plt.plot(numpy.arange(len(ys)) + 1, experiment_data['all']['answers_density'][group_name], color='black', label='Empirical distribution')
        plt.xlabel('Number of initiated series (groups of 10 answers)')
        plt.legend(loc=1, frameon=True)
        _savefig(filename, 'progress_all')
        plt.close()

        for group_name, group_data in sorted(experiment_data['all']['weibull'].items()):
            print group_name, group_data['params']


    if 'progress_milestones' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 5
        for i, (group_name, data) in enumerate(sorted(experiment_data['all']['progress_milestones'].items())):
            plt.plot(range(11, 11 + 10 * len(data), 10), map(lambda x: x['value'], data), label=_format_group_name(group_name), color=COLORS[i], marker=MARKERS[i])
        plt.legend(loc=4, frameon=True)
        plt.title('Stay curve (milestones)')
        plt.xlabel('Number of attempts')
        _savefig(filename, 'progress_milestones_all')
        plt.close()

    if 'output_rolling_success' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 5
        for i, (group_name, data) in enumerate(sorted(experiment_data['all']['output_rolling_success'].items())):
            to_plot = zip(*sorted(data.items()))
            plt.plot(to_plot[0], to_plot[1], label=_format_group_name(group_name), color=COLORS[i], marker=MARKERS[i])
        plt.legend(loc=2, frameon=True)
        plt.title('Output rolling success')
        plt.xlabel('Rolling success')
        plt.ylabel('Probability')
        _savefig(filename, 'output_rolling_success')
        plt.close()

    if 'stay_on_rolling_success' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 5
        for i, (group_name, data) in enumerate(sorted(experiment_data['all']['stay_on_rolling_success'].items())):
            to_plot = zip(*sorted(data.items()))
            plt.plot(to_plot[0], map(lambda x: x['value'], to_plot[1]), label=_format_group_name(group_name), color=COLORS[i], marker=MARKERS[i])
        plt.legend(loc=4, frameon=True)
        plt.title('Stay vs. Rolling success')
        plt.xlabel('Rolling success')
        plt.ylabel('Probability to continue')
        _savefig(filename, 'stay_on_rolling_success')
        plt.close()

    if 'attrition_bias' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 5
        for i, (group_name, data) in enumerate(sorted(experiment_data['all']['attrition_bias'].items())):
            plt.plot(range(len(data)), map(lambda x: x['value'], data), label=_format_group_name(group_name), color=COLORS[i], marker=MARKERS[i])
            plt.fill_between(
                range(len(data)),
                map(lambda x: x['confidence_interval_min'], data),
                map(lambda x: x['confidence_interval_max'], data),
                color=COLORS[i], alpha=0.35
            )
        plt.legend(loc=2, frameon=True, ncol=2)
        plt.title('Attrition bias')
        plt.xlabel('Minimal number of reference attempts')
        plt.ylabel('First attempt error')
        _savefig(filename, 'attrition_bias')
        plt.close()

    plt.close()
    if 'test_questions_hist' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 5
        for i, (group_name, data) in enumerate(sorted(experiment_data['all']['test_questions_hist'].items())):
            length = max(data.keys())
            for i in range(int(length)):
                if i not in data:
                    data[i] = 0
            to_plot = zip(*sorted(data.items()))
            plt.plot(to_plot[0], to_plot[1], label=_format_group_name(group_name))
        plt.legend(loc=1, frameon=True)
        _savefig(filename, 'test_questions')
        plt.close()

    if 'response_time_curve_correct_all' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 5
        plot_line(experiment_data['all']['response_time_curve_correct_all'], with_confidence=False)
        plt.xlabel('Attempt')
        plt.ylabel('Response time (ms)')
        plt.legend(loc=1, frameon=True)
        _savefig(filename, 'response_time_correct_all')
        plt.close()

    if 'response_time_curve_wrong_all' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 5
        plot_line(experiment_data['all']['response_time_curve_wrong_all'], with_confidence=False)
        plt.xlabel('Attempt')
        plt.ylabel('Response time (ms)')
        plt.title('All Learners')
        plt.legend(loc=1, frameon=True)
        _savefig(filename, 'response_time_wrong_all')
        plt.close()

    if 'error_hist' in experiment_data.get('all', {}) and 'learning_curve_global' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 10
        plt.subplot(211)
        to_plot_list = []
        for group_name, data in experiment_data['all']['error_hist'].iteritems():
            for _bin, _hist in zip(data['bins'], data['hist']):
                to_plot_list.append({'condition': group_name, 'bin': _bin, 'hist': _hist})
        to_plot = pandas.DataFrame(to_plot_list)
        sns.barplot(x='bin', y='hist', hue='condition', data=to_plot)
        plt.xticks(range(len(data['bins']) - 1), map(lambda x: '{} - {}'.format(int(100 * x), int(100 * (x + 0.1))), data['bins'][:-1]), rotation=30)
        plt.ylabel('Number of learners')
        plt.xlabel('Error rate (%)')
        plt.legend(loc=1, frameon=True, ncol=2)

        plt.subplot(212)
        ylim_learning_curve()
        plot_line(experiment_data['all']['learning_curve_global'], markevery=5, with_confidence=False)
        plt.legend(loc=1, frameon=True, ncol=2)
        plt.ylabel('Error rate')
        plt.xlabel('Attempt')
        _savefig(filename, 'success')
        plt.close()

    contexts_to_plot = zip(*sorted(experiment_data.get('contexts', {}).items(), key=lambda (_, val): -val['meta_all']['answers'])[:(context_cols * context_rows)])
    contexts_to_plot[0] = map(_format_context, contexts_to_plot[0])
    contexts_to_plot = zip(*contexts_to_plot)
    if len(contexts_to_plot) == context_cols * context_rows:
        if 'learning_curve_reverse' in experiment_data.get('all', {}) and 'learning_curve_all_reverse' in experiment_data.get('all', {}):
            rcParams['figure.figsize'] = 15, 20
            for i, (context, data) in enumerate(contexts_to_plot, start=1):
                plt.subplot(4, 2, 2 * i - 1)
                for j, (group_name, group_data) in enumerate(sorted(data['learning_curve_all_reverse'].items())):
                    plt.plot(range(len(group_data)), map(lambda x: x['value'], group_data), label=_format_group_name(group_name), marker=MARKERS[j], color=COLORS[j])
                    plt.fill_between(
                        range(len(group_data)),
                        map(lambda x: x['confidence_interval_min'], group_data),
                        map(lambda x: x['confidence_interval_max'], group_data),
                        color=COLORS[j], alpha=0.35
                    )
                plt.title('{}, all'.format(context))

                plt.subplot(4, 2, 2 * i)
                for j, (group_name, group_data) in enumerate(sorted(data['learning_curve_reverse'].items())):
                    plt.plot(range(len(group_data)), map(lambda x: x['value'], group_data), label=_format_group_name(group_name), marker=MARKERS[j], color=COLORS[j])
                    plt.fill_between(
                        range(len(group_data)),
                        map(lambda x: x['confidence_interval_min'], group_data),
                        map(lambda x: x['confidence_interval_max'], group_data),
                        color=COLORS[j], alpha=0.35
                    )
                plt.title('{}, filtered'.format(context))
                if i == 1:
                    plt.legend(loc=4, frameon=True)
            _savefig(filename, 'learning_curve_contexts_reverse')
            plt.close()

        if 'learning_curve' in experiment_data.get('all', {}) and 'learning_curve_all' in experiment_data.get('all', {}):
            rcParams['figure.figsize'] = 7.5 * context_cols, 5 * context_rows
            for i, (context, data) in enumerate(contexts_to_plot, start=1):
                plt.subplot(context_rows, context_cols, i)
                plot_line(data['learning_curve_all'], with_confidence=False)
                plt.title('{}, all'.format(context))
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.ylim(0, 0.8)
                if i % context_cols != 1:
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                else:
                    plt.ylabel('Error rate')
                if i <= context_cols * (context_rows - 1):
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                else:
                    plt.xlabel('Attempt')
                if i == 1:
                    plt.legend(loc=1, frameon=True, ncol=2)
            _savefig(filename, 'learning_curve_contexts')
            plt.close()

        if 'response_time_curve_correct' in experiment_data.get('all', {}) and 'response_time_curve_correct_all' in experiment_data.get('all', {}):
            rcParams['figure.figsize'] = 15, 20
            for i, (context, data) in enumerate(contexts_to_plot, start=1):
                plt.subplot(4, 2, 2 * i - 1)
                plot_line(data['response_time_curve_correct_all'], with_confidence=False)
                plt.title('{}, all'.format(context))

                plt.subplot(4, 2, 2 * i)
                plot_line(data['response_time_curve_correct'], with_confidence=False)
                plt.title('{}, filtered'.format(context))
                if i == 1:
                    plt.legend(loc=1, frameon=True)
            _savefig(filename, 'response_time_curve_contexts')
            plt.close()

        if 'item_occurences' in experiment_data.get('all', {}):
            sns_style = dict(SNS_STYLE)
            sns_style['font_scale'] = 2.5
            sns.set(**sns_style)
            rcParams['figure.figsize'] = 7.5 * context_cols, 5 * context_rows
            for i, (context, data) in enumerate(contexts_to_plot, start=1):
                ax = plt.subplot(context_rows, context_cols, i)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                for j, (group_name, group_data) in enumerate(sorted(data['item_occurences'].items())):
                    plt.title('{}'.format(context))
                    xs = range(len(group_data))
                    ys = map(lambda x: x['value'], sorted(group_data.values(), key=lambda x: x['difficulty']))
                    markevery=len(group_data) / 5
                    plt.plot(xs, ys, label=_format_group_name(group_name), color=COLORS[j], linewidth=5)
                    plt.xlim(0, len(group_data) - 1)
                    if i > context_cols * (context_rows - 1):
                        plt.xlabel('Items sorted by difficulty')
                    if i % context_cols == 1:
                        plt.ylabel('First occurence (median)')
                if i == 1:
                    plt.legend(loc=1, frameon=True, ncol=2, columnspacing=0.5, labelspacing=0.2, borderaxespad=0.1)
            _savefig(filename, 'item_occurences_order')
            plt.close()
            sns.set(**SNS_STYLE)

            rcParams['figure.figsize'] = 7.5 * context_cols, 5 * context_rows
            for i, (context, data) in enumerate(contexts_to_plot, start=1):
                ax = plt.subplot(context_rows, context_cols, i)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                for j, (group_name, group_data) in enumerate(sorted(data['item_occurences'].items())):
                    plt.title('{}'.format(context))
                    ys, xs = zip(*map(lambda x: (x['value'], _sigmoid(- x['difficulty'])), sorted(group_data.values(), key=lambda x: - x['difficulty'])))
                    plt.plot(xs, ys, label=_format_group_name(group_name), color=COLORS[1], markersize=MARKER_SIZE, linestyle=LINES[j], marker=MARKERS[j], markevery=5)
                    plt.xlim(0, 1)
                    if i > context_cols * (context_rows - 1):
                        plt.xlabel('Items sorted by difficulty')
                    if i % context_cols == 1:
                        plt.ylabel('First occurence (median)')
                if i == 1:
                    plt.legend(loc=2, frameon=True, ncol=2)
            _savefig(filename, 'item_occurences_difficulty')
            plt.close()

        if 'item_frequencies' in experiment_data.get('all', {}):
            rcParams['figure.figsize'] = 7.5 * context_cols, 5 * context_rows
            for i, (context, data) in enumerate(contexts_to_plot, start=1):
                ax = plt.subplot(context_rows, context_cols, i)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                for j, (group_name, group_data) in enumerate(sorted(data['item_frequencies'].items())):
                    plt.title('{}'.format(context))
                    xs = range(len(group_data))
                    ys = map(lambda x: x['value'], sorted(group_data.values(), key=lambda x: x['difficulty']))
                    plt.plot(xs, ys, label=_format_group_name(group_name), color=COLORS[1], markersize=MARKER_SIZE, marker=MARKERS[j], linestyle=LINES[j], markevery=5)
                    plt.xlim(0, len(group_data) - 1)
                    if i > context_cols * (context_rows - 1):
                        plt.xlabel('Items sorted by difficulty')
                    if i % context_cols == 1:
                        plt.ylabel('Frequency')
                if i == 1:
                    plt.legend(loc=9, frameon=True, ncol=2)
            _savefig(filename, 'item_frequencies_order')
            plt.close()

            rcParams['figure.figsize'] = 7.5 * context_cols, 5 * context_rows
            for i, (context, data) in enumerate(contexts_to_plot, start=1):
                ax = plt.subplot(context_rows, context_cols, i)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                for j, (group_name, group_data) in enumerate(sorted(data['item_frequencies'].items())):
                    plt.title('{}'.format(context))
                    ys, xs = zip(*map(lambda x: (x['value'], _sigmoid(- x['difficulty'])), sorted(group_data.values(), key=lambda x: - x['difficulty'])))
                    plt.plot(xs, ys, label=_format_group_name(group_name), color=COLORS[1], markersize=MARKER_SIZE, marker=MARKERS[j], linestyle=LINES[j], markevery=5)
                    plt.xlim(0, 1)
                    if i > context_cols * (context_rows - 1):
                        plt.xlabel('Items sorted by difficulty')
                    if i % context_cols == 1:
                        plt.ylabel('Frequency')
                if i == 1:
                    plt.legend(loc=2, frameon=True, ncol=2)
            _savefig(filename, 'item_frequencies_difficulty')
            plt.close()

        if 'attrition_bias' in experiment_data.get('all', {}):
            rcParams['figure.figsize'] = 7.5 * context_cols, 5 * context_rows
            for i, (context, data) in enumerate(contexts_to_plot, start=1):
                ax = plt.subplot(context_rows, context_cols, i)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plot_line(data['attrition_bias'], with_confidence=False)
                plt.title('{}'.format(context))
                if i == 1:
                    plt.legend(loc=2, frameon=True, ncol=2, )
                if i > context_cols * (context_rows - 1):
                    plt.xlabel('Minimal number of reference attempts')
                if i % context_cols == 1:
                    plt.ylabel('First attempt error')
            _savefig(filename, 'attrition_bias_contexts')
            plt.close()

    if 'learning_points' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 25, 25
        groups = []

        def _get_data(to_plot):
            return numpy.array([map(lambda x: x if numpy.isfinite(x) else 0, zip(*sorted(xs.items()))[1]) for (_, xs) in sorted(to_plot.items())])

        def _plot_learning_surface(ax, data, title, xticklabels, cmap=plt.cm.RdYlGn, vmin=0, vmax=1):
            plt.title(title)
            ax.set_xticks(numpy.arange(len(data)) + 0.5,  minor=False)
            ax.set_xticklabels(map(_format_time, xticklabels),  minor=False, rotation=90)
            pcolor = ax.pcolor(data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(pcolor)

        for i, (group_name, data) in enumerate(sorted(experiment_data['all']['learning_points'].items())):
            to_plot = defaultdict(lambda: {})
            for row in data:
                to_plot[row[0]][row[1]] = numpy.nan if row[2] is None else row[2]
            groups.append((group_name, to_plot))
        for i, (group_name, to_plot) in enumerate(groups):
            ax = plt.subplot(len(groups) + 1, len(groups) + 1, i + 2)
            _plot_learning_surface(ax, _get_data(to_plot), group_name, sorted(to_plot[0]))
            ax = plt.subplot(len(groups) + 1, len(groups) + 1, (len(groups) + 1) * (i + 1) + 1)
            _plot_learning_surface(ax, _get_data(groups[i][1]), groups[i][0], sorted(to_plot[0]))
            for j, (_, to_plot2) in enumerate(groups):
                ax = plt.subplot(len(groups) + 1, len(groups) + 1, (len(groups) + 1) * (i + 1) + j + 2)
                _plot_learning_surface(ax, _get_data(to_plot) - _get_data(to_plot2), '', sorted(to_plot[0]), cmap=plt.cm.RdYlGn, vmin=-0.2, vmax=0.2)

        _savefig(filename, 'learning_points_all')
        plt.close()

        rcParams['figure.figsize'] = 7.5, 5
        for i, (group_name, to_plot) in enumerate(groups):
            ax = plt.gca(projection='3d')
            data = _get_data(to_plot)
            xs = numpy.arange(len(data[0]))
            ys = numpy.arange(len(data))
            xs, ys = numpy.meshgrid(xs, ys)
            surface = ax.plot_surface(xs, ys, data, rstride=1, cstride=1, cmap=plt.cm.RdYlGn, linewidth=0, antialiased=False)
            ax.set_zlim(0, 1)
            plt.colorbar(surface)
            ax.set_xticklabels(map(_format_time, sorted(to_plot[0])),  minor=False, rotation=90)
            plt.title(group_name)
            _savefig(filename, 'learning_surface_{}'.format(group_name))
            plt.close()

    if 'contexts' in experiment_data and 'learning_curve_all' in experiment_data.get('all', {}):
        contexts, answer_nums = zip(*sorted(map(lambda x: (x[0], x[1]['meta_all']['answers']), experiment_data['contexts'].iteritems()), key=lambda x: -x[1]))
        error = numpy.array(map(lambda c: experiment_data['contexts'][c]['learning_curve_all'][50][0]['value'], contexts[:10]))
        contexts = map(_format_context, contexts[:10])
        ax = plt.subplot(111)
        plt.bar(range(len(answer_nums[:10])), map(lambda x: round(100 * x / float(sum(answer_nums))), answer_nums[:10]))
        plt.title('Top 10 mostly practiced contexts')
        plt.ylabel('Number of answers (%)')
        ax.set_xticks(numpy.arange(len(answer_nums[:10])),  minor=False)
        ax.set_xticklabels(contexts[:10],  minor=False, rotation=60)
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.xaxis.grid(False)
        ax2.set_ylabel('First attempt error', color=COLORS[2])
        for tl in ax2.get_yticklabels():
            tl.set_color(COLORS[2])
        ax2.yaxis.grid(False)
        ax2.scatter(numpy.arange(len(error)) + 0.4, error, color=COLORS[2], marker=MARKERS[2], s=10 * MARKER_SIZE)
        _savefig(filename, 'context_answers')
        plt.close()

        contexts_to_plot = ['Europe, state', 'Czech Rep., mountains', 'Czech Rep., city']
        rcParams['figure.figsize'] = len(contexts_to_plot) * 7.5, 5
        for i, context in enumerate(contexts_to_plot):
            plt.subplot(1, len(contexts_to_plot), i + 1)
            plot_line(experiment_data['contexts'][context]['learning_curve_all'], with_confidence=True)
            plt.ylim(0, 1)
            plt.title(context)
            if i == 0:
                plt.ylabel('All learners \nError rate')
                plt.legend(loc=1, frameon=True)
            plt.xlabel('Attempt')
        _savefig(filename, 'context_comparison')
        plt.close()

    if 'contexts' in experiment_data and 'practice_error' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 6
        contexts, answer_nums, practice_error = zip(*sorted(map(
            lambda x: (x[0], x[1]['meta_all']['answers'], x[1]['practice_error']),
            experiment_data['contexts'].iteritems()
        ), key=lambda x: -x[1]))
        practice_error_plot = defaultdict(list)
        for x in practice_error:
            for key, val in x.iteritems():
                if len(practice_error_plot[key]) < 10:
                    practice_error_plot[key].append(val)
        contexts = map(_format_context, contexts[:10])
        ax = plt.subplot(111)
        plt.bar(range(len(answer_nums[:10])), map(lambda x: round(100 * x / float(sum(answer_nums))), answer_nums[:10]), color=COLORS[0])
        plt.title('Top 10 mostly practiced contexts')
        ax.set_xticks(numpy.arange(len(answer_nums[:10])),  minor=False)
        ax.set_xticklabels(contexts[:10],  minor=False, rotation=60)
        ax.set_ylim(0, 20)
        ax2 = ax.twinx()
        ax2.set_ylim(0, 0.7)
        ax2.xaxis.grid(False)
        ax2.set_ylabel('Error rate\n(without reference answers)')
        ax.set_ylabel('Number of answers (%)', color=COLORS[2])
        for tl in ax.get_yticklabels():
            tl.set_color(COLORS[2])
        ax2.yaxis.grid(False)
        plot_line(practice_error_plot, xs=numpy.arange(10) + 0.4, with_confidence=False, ax=ax2, skip_line=True)
        ax2.legend(loc=1, frameon=True, ncol=2)
        plt.xlim(0, 10)
        _savefig(filename, 'context_practice_error')
        plt.close()

    if 'contexts' in experiment_data and 'learning_gain' in experiment_data.get('all', {}):
        contexts_to_plot = zip(*sorted(experiment_data.get('contexts', {}).items(), key=lambda (_, val): -val['meta_all']['answers'])[:10])
        contexts_to_plot[0] = map(_format_context, contexts_to_plot[0])
        context_order = dict(zip(contexts_to_plot[0], range(len(contexts_to_plot[0]))))
        contexts_to_plot = zip(*contexts_to_plot)
        to_plot = defaultdict(lambda: dict())
        for context, data in contexts_to_plot:
            for group_name, group_data in data['learning_gain'].iteritems():
                to_plot[group_name][context] = group_data

        ax = plt.gca()
        for i, (group_name, group_data) in enumerate(sorted(to_plot.items())):
            contexts, values = zip(*sorted(group_data.items(), key=lambda x: context_order[x[0]]))
            confidence_vals = map(lambda x: (x['value'] - x['confidence_interval_min'], x['confidence_interval_max'] - x['value']), values)
            values = map(lambda x: x['value'], values)
            ax.bar(
                numpy.arange(len(group_data)) + i * 0.2,
                values, 0.2,
                yerr=zip(*confidence_vals), error_kw={'ecolor': 'black'},
                color=COLORS[i], hatch=HATCHES[i], label=_format_group_name(group_name),
            )
            ax.set_xticks(numpy.arange(len(group_data)) + 0.4)
            ax.set_xticklabels(contexts, minor=False, rotation=60)
            ax.set_ylabel('Learning gain')
        ax.legend(loc=3, frameon=True, ncol=2)
        _savefig(filename, 'context_learning_gain')
        plt.close()

    if 'contexts' in experiment_data and 'learning_slope' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 15, 5

        gs = GridSpec(2, 2, width_ratios=[2, 3], height_ratios=[4, 1], hspace=0)
        plt.subplot(gs.new_subplotspec((0, 0), rowspan=2))
        ylim_learning_curve()
        plt.title('Learning curve')
        plot_line(experiment_data['all']['learning_curve_fit_all'], with_confidence=False, marker=False, linewidth=5)
        plt.xlabel('Attempt')
        plt.ylabel('Error rate')
        plt.legend(loc=1, frameon=True, ncol=2)

        contexts_to_plot = zip(*sorted(experiment_data.get('contexts', {}).items(), key=lambda (_, val): -val['meta_all']['answers'])[:10])
        contexts_to_plot[0] = map(_format_context, contexts_to_plot[0])
        context_order = dict(zip(contexts_to_plot[0], range(len(contexts_to_plot[0]))))
        contexts_to_plot = zip(*contexts_to_plot)
        to_plot = defaultdict(lambda: dict())
        for context, data in contexts_to_plot:
            for group_name, group_data in data['learning_slope'].iteritems():
                to_plot[group_name][context] = group_data

        ax = plt.subplot(gs[1])
        for i, (group_name, group_data) in enumerate(sorted(to_plot.items())):
            contexts, values = zip(*sorted(group_data.items(), key=lambda x: context_order[x[0]]))
            confidence_vals = map(lambda x: (x['value'] - x['confidence_interval_min'], x['confidence_interval_max'] - x['value']), values)
            values = map(lambda x: x['value'], values)
            ax.bar(
                numpy.arange(len(group_data)) + i * 0.2,
                values, 0.2,
                yerr=zip(*confidence_vals), error_kw={'ecolor': 'black'},
                color=COLORS[i], hatch=HATCHES[i], label=_format_group_name(group_name),
            )
            ax.set_xticks(numpy.arange(len(group_data)) + 0.4)
            ax.set_xticklabels(contexts, minor=False, rotation=60)
            ax.set_ylabel('k')
            ax.set_title('Learning rate')
        ax.legend(loc=3, frameon=True, ncol=2, columnspacing=0.4, labelspacing=0, borderaxespad=0)
        _savefig(filename, 'context_learning_slope')
        plt.close()

    if 'weibull' in experiment_data.get('all', {}) and 'answers_density' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 15, 10
        for i, (group_name, group_data) in enumerate(experiment_data['all']['weibull'].iteritems()):
            plt.subplot(2, 2, i + 1)
            ys = map(lambda x: x['value'], group_data['serie'])
            plt.plot(numpy.arange(len(ys)) + 0.5, ys)
            plt.plot(numpy.arange(len(ys)) + 0.5, experiment_data['all']['answers_density'][group_name], color='black')
            plt.title(group_name)
        _savefig(filename, 'weibull')
        plt.close()

    if 'contexts' in experiment_data and 'attrition_bias' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 15, 5
        contexts_to_plot = ['Europe, state', 'Asia, state']
        for i, context_name in enumerate(contexts_to_plot, start=1):
            data = experiment_data['contexts'][context_name]
            ax = plt.subplot(1, 2, i)
            plot_line(data['attrition_bias'], with_confidence=False, marker=False, linewidth=5)
            plt.title('{}'.format(context_name))
            if i == 1:
                plt.legend(loc=2, frameon=True, ncol=2)
            plt.xlabel('Minimal number of reference attempts')
            plt.ylabel('First attempt error')
        _savefig(filename, 'attrition_bias_contexts_selected')
        plt.close()

    if 'learning_points_all' in experiment_data.get('all', {}):
        rcParams['figure.figsize'] = 7.5, 5
        to_plot = defaultdict(lambda: {})
        for row in experiment_data['all']['learning_points_all']:
            if row[0] == 0:
                continue
            to_plot[row[0]][row[1]] = numpy.nan if row[2] is None else 1 - row[2]
        ax = plt.gca(projection='3d')
        data = _get_data(to_plot)
        xs = numpy.arange(len(data[0]))[::-1]
        ys = (numpy.arange(len(data)) + 1)[::-1]
        xs, ys = numpy.meshgrid(xs, ys)
        surface = ax.plot_surface(xs, ys, data, rstride=1, cstride=1, cmap=plt.cm.RdYlGn_r, linewidth=0, antialiased=False)
        ax.set_zlim(0, 0.5)
        plt.colorbar(surface)
        ax.set_xticklabels(map(_format_time, sorted(to_plot[1].keys(), reverse=True)),  minor=False, rotation=90)
        ax.set_yticklabels(numpy.arange(len(data))[::-1] + 2, minor=False)
        plt.title('Learning surface')
        ax.set_xlabel('\n\nTime')
        ax.set_ylabel('\nAttempt')
        ax.set_zlabel('\nError rate')
        _savefig(filename, 'learning_surface_all'.format(group_name))
        plt.close()


plot_experiment_data(pa.get_experiment_data(
    'ab_random_random_3',
    compute_experiment_data,
    'experiment_cache', cached=True, debug=True,
    answer_limit=1, curve_length=10, progress_length=100, density_length=300, contexts=True,
    filter_invalid_response_time=False,
    #context_name='Europe', term_type='state',
    #keys=['learning_curve_fit'] , bootstrap_samples=100
    keys=[
        'item_occurences', 'learning_points', 'learning_points_all', 'learning_gain',
        'learning_curve', 'learning_curve_fit', 'learning_curve_all', 'learning_slope',
        'learning_curve_fit_all', 'learning_curve_reverse',
        'learning_curve_fit_reverse', 'attrition_bias', 'practice_error',
    ]#, bootstrap_samples=100
    #keys=['progress', 'weibull', 'answers_density'], bootstrap_samples=1000
), 'target_difficulty')
