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


SNS_STYLE = {'style': 'white', 'font_scale': 1.8}
sns.set(**SNS_STYLE)


SETUP = {
    6: 'random-adaptive',
    7: 'random-random',
    8: 'adaptive-adaptive',
    9: 'adaptive-random',
}

MARKES = "dos^"


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


def load_data(answer_limit):
    answers = pandas.read_csv('./answers.csv', index_col=False, parse_dates=['time'])
    flashcards = pandas.read_csv('./flashcards.csv', index_col=False)

    answers['experiment_setup_name'] = answers['experiment_setup_id'].apply(lambda i: SETUP[i])

    valid_users = map(lambda x: x[0], filter(lambda x: x[1] >= answer_limit, answers.groupby('user_id').apply(len).to_dict().items()))
    answers = answers[answers['user_id'].isin(valid_users)]

    invalid_users = answers[answers['context_id'] == 17]['user_id'].unique()
    answers = answers[~answers['user_id'].isin(invalid_users)]

    invalid_users = answers[answers['response_time'] < 0]['user_id'].unique()
    answers = answers[~answers['user_id'].isin(invalid_users)]

    return pandas.merge(answers, flashcards, on='item_id', how='inner').sort(['user_id', 'id'])


def progress(data, length=60):
    user_answers = data.groupby('user_id').apply(len).to_dict().values()
    result = []
    for i in range(length):
        result.append(len(filter(lambda x: x > i, user_answers)) / float(len(user_answers)))
    return result


def learning_curve(data, length=10, user_length=None, context_answer_limit=100):

    def _learning_curve(group):
        if len(group) < context_answer_limit:
            return []
        user_answers_dict = defaultdict(list)
        for row in iterdicts(group):
            user_answers_dict[row['user_id']].append(row['item_asked_id'] == row['item_answered_id'])
        user_answers = [
            answers[:min(len(answers), length)] + [None for _ in range(length - min(len(answers), length))]
            for answers in user_answers_dict.itervalues()
            if user_length is None or len(answers) >= user_length
        ]

        def _mean_with_confidence(xs):
            return binomial_confidence_mean(filter(lambda x: x is not None, xs))
        curve = map(_mean_with_confidence, zip(*user_answers))
        return map(lambda x: x[0] - curve[0][0], curve)
    context_curves = filter(lambda x: len(x) != 0, data.groupby(['context_name', 'term_type']).apply(_learning_curve))
    return map(lambda xs: numpy.mean(filter(lambda x: not numpy.isnan(x), xs)), zip(*context_curves))


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

        def _median_with_confidence(xs):
            return confidence_median(filter(lambda x: x is not None, list(xs)))
        curve = map(_median_with_confidence, zip(*user_answers))
        return map(lambda x: x[0] - curve[0][0], curve)
    context_curves = filter(lambda x: len(x) != 0, data.groupby(['context_name', 'term_type']).apply(_response_time_curve))
    return map(lambda xs: numpy.mean(filter(lambda x: not numpy.isnan(x), xs)), zip(*context_curves))


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


def compute_experiment_data(term_type=None, term_name=None, context_name=None, answer_limit=10, curve_length=5, progress_length=60):
    data = load_data(answer_limit)
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

    def _group_experiment_data(data_all):
        data = data_all[data_all['metainfo_id'] == 1]
        groupped = data.groupby('experiment_setup_name')
        groupped_all = data_all.groupby('experiment_setup_name')
        return {
            'learning_points': groupped.apply(lambda g: learning_points(g, length=curve_length)).to_dict(),
            'learning_curve_all': groupped.apply(lambda g: learning_curve(g, length=curve_length)).to_dict(),
            'learning_curve': groupped.apply(lambda g: learning_curve(g, length=curve_length, user_length=curve_length)).to_dict(),
            'response_time_curve_all': groupped.apply(lambda g: response_time_curve(g, length=curve_length)).to_dict(),
            'response_time_curve': groupped.apply(lambda g: response_time_curve(g, length=curve_length, user_length=curve_length)).to_dict(),
            'progress': groupped_all.apply(lambda g: progress(g, length=progress_length)).to_dict(),
            'meta': groupped_all.apply(meta).to_dict(),
            'meta_all': meta(data_all),
        }
    return {
        'all': _group_experiment_data(data),
        'contexts': {
            '{}, {}'.format(context_name, term_type): value
            for ((context_name, term_type), value) in data.groupby(['context_name', 'term_type']).apply(_group_experiment_data).to_dict().iteritems()
        },
    }


def plot_experiment_data(experiment_data, filename):
    rcParams['figure.figsize'] = 15, 5
    plt.subplot(121)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['learning_curve_all'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKES[i])
    plt.title('All Users')

    plt.subplot(122)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['learning_curve'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKES[i])
    plt.title('Filtered Users')
    plt.legend(loc=4, frameon=True)
    plt.savefig('{}_learning_curve_all.svg'.format(filename))
    plt.close()

    rcParams['figure.figsize'] = 15, 10
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['progress'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKES[i])
    plt.legend(loc=1, frameon=True)
    plt.savefig('{}_progress_all.svg'.format(filename))
    plt.close()

    rcParams['figure.figsize'] = 15, 5
    plt.subplot(121)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['response_time_curve_all'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKES[i])
    plt.title('All Users')

    plt.subplot(122)
    for i, (group_name, data) in enumerate(sorted(experiment_data['all']['response_time_curve'].items())):
        plt.plot(range(len(data)), data, label=group_name, marker=MARKES[i])
    plt.title('Filtered Users')
    plt.legend(loc=1, frameon=True)
    plt.savefig('{}_learning_curve_all.svg'.format(filename))
    plt.close()

    contexts_to_plot = sorted(experiment_data['contexts'].items(), key=lambda (_, val): -val['meta_all']['answers'])[:4]
    if len(contexts_to_plot) == 4:
        rcParams['figure.figsize'] = 15, 20
        for i, (context, data) in enumerate(contexts_to_plot, start=1):
            plt.subplot(4, 2, 2 * i - 1)
            for j, (group_name, group_data) in enumerate(sorted(data['learning_curve_all'].items())):
                plt.plot(range(len(group_data)), group_data, label=group_name, marker=MARKES[j])
            plt.title('{}, all'.format(context))

            plt.subplot(4, 2, 2 * i)
            for j, (group_name, group_data) in enumerate(sorted(data['learning_curve'].items())):
                plt.plot(range(len(group_data)), group_data, label=group_name, marker=MARKES[j])
            plt.title('{}, filtered'.format(context))
            if i == 1:
                plt.legend(loc=4, frameon=True)
        plt.savefig('{}_learning_curve_contexts.svg'.format(filename))
        plt.close()

        rcParams['figure.figsize'] = 15, 20
        for i, (context, data) in enumerate(contexts_to_plot, start=1):
            plt.subplot(4, 2, 2 * i - 1)
            for j, (group_name, group_data) in enumerate(sorted(data['response_time_curve_all'].items())):
                plt.plot(range(len(group_data)), group_data, label=group_name, marker=MARKES[j])
            plt.title('{}, all'.format(context))

            plt.subplot(4, 2, 2 * i)
            for j, (group_name, group_data) in enumerate(sorted(data['response_time_curve'].items())):
                plt.plot(range(len(group_data)), group_data, label=group_name, marker=MARKES[j])
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
    answer_limit=1, progress_length=100,
), 'random_random')
