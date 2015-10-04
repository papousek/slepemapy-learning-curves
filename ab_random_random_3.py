# works with:
#   http://data-private.slepemapy.cz/ab-experiment-random-parts-3.zip

import proso.analysis as pa
import pandas
import numpy
from proso.geography.dfutil import iterdicts
from collections import defaultdict
from proso.metric import binomial_confidence_mean, confidence_median, confidence_value_to_json
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import scikits.bootstrap as bootstrap


SNS_STYLE = {'style': 'white', 'font_scale': 1.8}
sns.set(**SNS_STYLE)


SETUP = {
    6: 'random-adaptive',
    7: 'random-random',
    8: 'adaptive-adaptive',
    9: 'adaptive-random',
}

MARKERS = "dos^"
COLORS = sns.color_palette()


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


def load_data(answer_limit, filter_invalid_tests=True):
    answers = pandas.read_csv('./answers.csv', index_col=False, parse_dates=['time'])
    flashcards = pandas.read_csv('./flashcards.csv', index_col=False)

    answers['experiment_setup_name'] = answers['experiment_setup_id'].apply(lambda i: SETUP[i])

    valid_users = map(lambda x: x[0], filter(lambda x: x[1] >= answer_limit, answers.groupby('user_id').apply(len).to_dict().items()))
    answers = answers[answers['user_id'].isin(valid_users)]

    invalid_users = answers[answers['context_id'] == 17]['user_id'].unique()
    answers = answers[~answers['user_id'].isin(invalid_users)]

    invalid_users = answers[answers['response_time'] < 0]['user_id'].unique()
    answers = answers[~answers['user_id'].isin(invalid_users)]

    if filter_invalid_tests:
        invalid_users = set()
        last_user = None
        last_context = None
        counter = None
        for row in iterdicts(answers.sort(['user_id', 'context_id', 'id'])):
            if last_user != row['user_id'] or last_context != row['context_id']:
                last_user = row['user_id']
                last_context = row['context_id']
                counter = 0
            if row['metainfo_id'] == 1 and counter % 10 != 0:
                invalid_users.add(row['user_id'])
            counter += 1
        answers = answers[~answers['user_id'].isin(invalid_users)]

    return pandas.merge(answers, flashcards, on='item_id', how='inner').sort(['user_id', 'id'])


def progress(data, length=60, with_confidence=False):
    user_answers = data.groupby('user_id').apply(len).to_dict().values()

    def _progress_confidence(i):
        xs = map(lambda x: x > i, user_answers)
        value = numpy.mean(xs)
        if with_confidence:
            confidence = bootstrap.ci(xs, numpy.mean, method='pi')
        else:
            confidence = (value, value)
        return {
            'value': value,
            'confidence_interval_min': confidence[0],
            'confidence_interval_max': confidence[1],
        }

    result = []
    for i in range(length):
        result.append(_progress_confidence(i))
    return result


def learning_curve(data, length=10, user_length=None, context_answer_limit=100, normalize=False, reverse=False):

    def _learning_curve(group):
        if len(group) < context_answer_limit:
            return []
        user_answers_dict = defaultdict(list)
        for row in iterdicts(group):
            user_answers_dict[row['user_id']].append(row['item_asked_id'] == row['item_answered_id'])
        if reverse:
            for user in user_answers_dict.keys():
                user_answers_dict[user] = user_answers_dict[user][::-1]
        user_answers = [
            answers[:min(len(answers), length)] + [None for _ in range(length - min(len(answers), length))]
            for answers in user_answers_dict.itervalues()
            if user_length is None or len(answers) >= user_length
        ]
        if reverse:
            user_answers = map(lambda xs: xs[::-1], user_answers)

        def _mean_with_numbers(xs):
            xs_filtered = filter(lambda x: x is not None, xs)
            return numpy.mean(xs_filtered), len(xs_filtered)
        curve = map(_mean_with_numbers, zip(*user_answers))
        if normalize:
            return map(lambda x: (x[0] - curve[0][0], x[1]), curve)
        else:
            return curve
    context_curves = filter(lambda x: len(x) != 0, data.groupby(['context_name', 'term_type']).apply(_learning_curve))

    def _weighted_mean(xs):
        xs = filter(lambda x: not numpy.isnan(x[0]), xs)
        return sum(map(lambda x: x[0] * x[1], xs)) / float(sum(map(lambda x: x[1], xs)))
    return map(_weighted_mean, zip(*context_curves))


def test_questions(data, length=100):
    last_user = None
    last_context = None
    counter = None
    result = defaultdict(lambda: 0)
    for row in iterdicts(data.sort(['user_id', 'context_id', 'id'])):
        if last_user != row['user_id'] or last_context != row['context_id']:
            last_user = row['user_id']
            last_context = row['context_id']
            counter = 0
        if row['metainfo_id'] == 1 and counter < length:
            result[counter] += 1
        counter += 1
    return dict(result.items())


def response_time_curve(data, length=10, user_length=None, context_answer_limit=100):

    def _response_time_curve(group):
        if len(group) < context_answer_limit:
            return []
        user_answers_dict = defaultdict(list)
        for row in iterdicts(group):
            user_answers_dict[row['user_id']].append(row['response_time'])
        user_answers = [
            answers[:min(len(answers), length)] + [None for _ in range(length - min(len(answers), length))]
            for answers in user_answers_dict.itervalues()
            if user_length is None or len(answers) >= user_length
        ]
        return user_answers

    all_answers = [ans for c_ans in data.groupby(['context_name', 'term_type']).apply(_response_time_curve).to_dict().values() for ans in c_ans]
    def _aggregate(xs):
        xs = filter(lambda x: x is not None, xs)
        return numpy.median(xs)
    return map(_aggregate, zip(*all_answers))


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


def compute_experiment_data(term_type=None, term_name=None, context_name=None, answer_limit=10, curve_length=5, progress_length=60, filter_invalid_tests=True, with_confidence=False):
    data = load_data(answer_limit, filter_invalid_tests=filter_invalid_tests)
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

    def _group_experiment_data(data_all, extended=False):
        data = data_all[data_all['metainfo_id'] == 1]
        groupped = data.groupby('experiment_setup_name')
        groupped_all = data_all.groupby('experiment_setup_name')
        result = {
            'learning_points': groupped.apply(lambda g: learning_points(g, length=curve_length)).to_dict(),
            'learning_curve_all': groupped.apply(lambda g: learning_curve(g, length=curve_length)).to_dict(),
            'learning_curve_all_reverse': groupped.apply(lambda g: learning_curve(g, length=curve_length, reverse=True)).to_dict(),
            'learning_curve': groupped.apply(lambda g: learning_curve(g, length=curve_length, user_length=curve_length)).to_dict(),
            'learning_curve_reverse': groupped.apply(lambda g: learning_curve(g, length=curve_length, user_length=curve_length, reverse=True)).to_dict(),
            'response_time_curve_all': groupped.apply(lambda g: response_time_curve(g, length=curve_length)).to_dict(),
            'response_time_curve': groupped.apply(lambda g: response_time_curve(g, length=curve_length, user_length=curve_length)).to_dict(),
            'test_questions_hist': groupped_all.apply(lambda g: test_questions(g, length=progress_length)).to_dict(),
            'meta': groupped_all.apply(meta).to_dict(),
            'meta_all': meta(data_all),
        }
        if extended:
            result['progress'] = groupped_all.apply(lambda g: progress(g, length=progress_length, with_confidence=with_confidence)).to_dict()
        return result
    return {
        'all': _group_experiment_data(data, extended=True),
        'contexts': {
            '{}, {}'.format(context_name, term_type): value
            for ((context_name, term_type), value) in data.groupby(['context_name', 'term_type']).apply(_group_experiment_data).to_dict().iteritems()
        },
    }


def plot_experiment_data(experiment_data, filename):
    rcParams['figure.figsize'] = 15, 5
    plt.subplot(121)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['learning_curve_all'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKERS[i])
    plt.title('All Users')

    plt.subplot(122)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['learning_curve'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKERS[i])
    plt.title('Filtered Users')
    plt.legend(loc=4, frameon=True)
    plt.savefig('{}_learning_curve_all.svg'.format(filename))
    plt.close()

    rcParams['figure.figsize'] = 15, 5
    plt.subplot(121)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['learning_curve_all_reverse'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKERS[i])
    plt.title('All Users')

    plt.subplot(122)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['learning_curve_reverse'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKERS[i])

    plt.title('Filtered Users')
    plt.legend(loc=4, frameon=True)
    plt.savefig('{}_learning_curve_all_reverse.svg'.format(filename))
    plt.close()

    rcParams['figure.figsize'] = 15, 5
    plt.subplot(121)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['progress'].items())):
        plt.plot(range(len(data)), map(lambda x: x['value'], data), label=group_name, color=COLORS[i])
    plt.legend(loc=1, frameon=True)
    plt.title('Stay curve')
    plt.xlabel('Number of attempts')

    plt.subplot(122)
    to_plot = zip(*sorted(experiment_data['all']['progress'].items()))
    to_plot[1] = map(lambda x: x[10], to_plot[1])
    xs = numpy.arange(len(to_plot[0]))
    ys = map(lambda d: d['value'], to_plot[1])
    errors = map(numpy.array, zip(*map(lambda d: [d['confidence_interval_min'], d['confidence_interval_max']], to_plot[1])))
    plt.ylim(0.6, 0.8)
    plt.bar(xs, ys, yerr=[ys - errors[0], errors[1] - ys], ecolor='black', error_kw={'linewidth': 4}, color=COLORS[:len(to_plot[0])])
    plt.gca().yaxis.grid(True)
    plt.xticks(xs + 0.35, to_plot[0], rotation=10)
    plt.title('Users having at least 11 answers')
    plt.xlabel('Variant of algorithm for question construction')
    plt.tight_layout()
    plt.savefig('{}_progress_all.png'.format(filename))
    plt.close()

    rcParams['figure.figsize'] = 7.5, 5
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['test_questions_hist'].items())):
        length = max(data.keys())
        for i in range(int(length)):
            if i not in data:
                data[i] = 0
        to_plot = zip(*sorted(data.items()))
        plt.plot(to_plot[0], to_plot[1], label=group_name)
    plt.legend(loc=1, frameon=True)
    plt.savefig('{}_test_questions.svg'.format(filename))
    plt.close()

    rcParams['figure.figsize'] = 15, 5
    plt.subplot(121)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['response_time_curve_all'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKERS[i])
    plt.title('All Users')

    plt.subplot(122)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['response_time_curve'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKERS[i])
    plt.title('Filtered Users')
    plt.legend(loc=1, frameon=True)
    plt.savefig('{}_response_time_all.svg'.format(filename))
    plt.close()

    contexts_to_plot = sorted(experiment_data['contexts'].items(), key=lambda (_, val): -val['meta_all']['answers'])[:4]
    if len(contexts_to_plot) == 4:
        rcParams['figure.figsize'] = 15, 20
        for i, (context, data) in enumerate(contexts_to_plot, start=1):
            plt.subplot(4, 2, 2 * i - 1)
            for j, (group_name, group_data) in enumerate(sorted(data['learning_curve_all'].items())):
                plt.plot(range(len(group_data)), group_data, label=group_name, marker=MARKERS[j])
            plt.title('{}, all'.format(context))

            plt.subplot(4, 2, 2 * i)
            for j, (group_name, group_data) in enumerate(sorted(data['learning_curve'].items())):
                plt.plot(range(len(group_data)), group_data, label=group_name, marker=MARKERS[j])
            plt.title('{}, filtered'.format(context))
            if i == 1:
                plt.legend(loc=4, frameon=True)
        plt.savefig('{}_learning_curve_contexts.svg'.format(filename))
        plt.close()

        rcParams['figure.figsize'] = 15, 20
        for i, (context, data) in enumerate(contexts_to_plot, start=1):
            plt.subplot(4, 2, 2 * i - 1)
            for j, (group_name, group_data) in enumerate(sorted(data['response_time_curve_all'].items())):
                plt.plot(range(len(group_data)), group_data, label=group_name, marker=MARKERS[j])
            plt.title('{}, all'.format(context))

            plt.subplot(4, 2, 2 * i)
            for j, (group_name, group_data) in enumerate(sorted(data['response_time_curve'].items())):
                plt.plot(range(len(group_data)), group_data, label=group_name, marker=MARKERS[j])
            plt.title('{}, filtered'.format(context))
            if i == 1:
                plt.legend(loc=1, frameon=True)
        plt.savefig('{}_response_time_curve_contexts.svg'.format(filename))
        plt.close()

    rcParams['figure.figsize'] = 10, 10
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['learning_points'].items())):
        to_plot = defaultdict(lambda: {})
        for row in data:
            to_plot[row[0]][row[1]] = numpy.nan if row[2] is None else row[2]
        ax = plt.subplot(2, 2, i)
        plt.title(group_name)
        ax.set_xticks(numpy.arange(len(to_plot[0])) + 0.5,  minor=False)
        ax.set_xticklabels(map(_format_time, sorted(to_plot[0])),  minor=False, rotation=90)
        pcolor = ax.pcolor(numpy.array([zip(*xs.items())[1] for (_, xs) in sorted(to_plot.items())]), cmap=plt.cm.RdYlGn, vmin=0, vmax=1)
        plt.colorbar(pcolor)
    plt.tight_layout()
    plt.savefig('{}_learning_points_all.png'.format(filename))
    plt.close()


plot_experiment_data(pa.get_experiment_data(
    'ab_random_random_3',
    compute_experiment_data,
    'experiment_cache', cached=True,
    answer_limit=1, progress_length=50,
    with_confidence=True
), 'random_random')
