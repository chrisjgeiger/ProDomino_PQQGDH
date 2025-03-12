from collections.abc import Mapping

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


class Hparams(Mapping):
    def __init__(self, params: dict):
        for k, v in params.items():
            if isinstance(v, dict):
                v = Hparams(v)
            self.__dict__[k] = v

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, k):
        return self.__dict__[k]

    def __iter__(self):
        return iter(self.__dict__)

    def __str__(self):
        out_str = ['\n\033[1mHyperparameter Overview:\033[0m\n\n']
        for k, v in self.__dict__.items():

            if isinstance(v, Hparams):
                out = v.__str__()
                out = out.replace('\n\033[1mHyperparameter Overview:\033[0m\n\n',
                                  f'\033[92m{k}\033[0m:')
                out_str.append(out.replace('\n', '\n\t'))

            else:
                out_str.append(f'\033[92m{k}\033[0m: {v}')

        return '\n'.join(out_str)

    def to_md(self):
        out_str = ['\n### Hyperparameter\n\n']
        for k, v in self.__dict__.items():

            if isinstance(v, Hparams):
                out = v.to_md()
                out = out.replace('\n### Hyperparameter\n\n',
                                  f'-  **{k}**')
                out_str.append(out.replace('\n', '\n\t'))

            else:
                out_str.append(f'-  **{k}**: {v}')

        return '\n'.join(out_str)

    def to_dict(self) -> dict:
        out_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Hparams):
                for k2, v2 in v.to_dict().items():
                    out_dict[f'{k}.{k2}'] = v2
            else:
                out_dict[k] = v
        return out_dict

    @classmethod
    def load_params(cls, path: str):
        with open(path, 'r') as f:
            config = Hparams(yaml.safe_load(f))
        return cls(config)


def parse_hparams(tree_dict, lvl, val):
    if len(lvl) == 1:
        tree_dict[lvl[0]] = val
    else:
        if tree_dict.get(lvl[0]) is None:
            tree_dict[lvl[0]] = {}
            tree_dict[lvl[0]] = parse_hparams(tree_dict[lvl[0]], lvl[1:], val)
        else:
            tree_dict[lvl[0]] = parse_hparams(tree_dict[lvl[0]], lvl[1:], val)
    return tree_dict


def make_performance_plot(data, std=1, save_path=None):
    metrics = ['metrics/precision', 'loss/train', 'loss/valid',
               'metrics/auroc']
    train_colors = [sns.color_palette("Paired")[i] for i in range(0, len(sns.color_palette("Paired")), 2)]
    valid_colors = [sns.color_palette("Paired")[i] for i in range(1, len(sns.color_palette("Paired")), 2)]
    fig = plt.figure(figsize=(12, 10), dpi=300)
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
    sns.lineplot(data=data[data.metric == metrics[-1]][:], x='step', y='value', hue='Name', ax=ax1,
                 palette=valid_colors)
    ax1.set_title('AUROC')
    ax1.set_ylim(-0.1, 1.1)

    ax1.get_legend().remove()

    ax2 = fig.add_subplot(gs[0, 1])  # row 0, col 1
    sns.lineplot(data=data[data.metric == metrics[0]][:], x='step', y='value', hue='Name', ax=ax2, palette=valid_colors)
    ax2.set_title('Precision')
    ax2.set_ylim(-0.1, 1.1)

    ax2.get_legend().remove()

    ax3 = fig.add_subplot(gs[1, 0])  # row 1, span all columns
    ax3.set_title('Loss on Train and Validation data')
    train_loss = data[data.metric == metrics[1]]
    valid_loss = data[data.metric == metrics[2]]
    train_loss['Name'] = train_loss['Name'].apply(lambda x: f'Train: {x}')
    valid_loss['Name'] = valid_loss['Name'].apply(lambda x: f'Valid: {x}')
    ax_lim = min([valid_loss[valid_loss.Name == name].step.max() for name in valid_loss.Name.unique()])
    sns.lineplot(data=train_loss, x='step', y='value', hue='Name', ax=ax3, palette=train_colors)
    sns.lineplot(data=valid_loss, x='step', y='value', hue='Name', ax=ax3,
                 # linestyle='--',
                 palette=valid_colors)
    # ax3.set_ylim(0,#data[data.metric==metrics[1]].value.mean() - 1/2*data[data.metric==metrics[1]].value.std(),
    #      data[data.metric==metrics[1]].value.mean() + std*data[data.metric==metrics[1]].value.std())

    # box = ax3.get_position()
    # ax3.set_position([box.x0, box.y0 + box.height * 0.1,
    #              box.width, box.height * 0.9])

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Steps')
        ax.set_xlim(0, ax_lim)
    fig.suptitle('Training Overview')

    fig.tight_layout()

    ax3.legend(loc='center left', bbox_to_anchor=(1.1, 0.5),
               fancybox=True, shadow=False, ncol=1)
    if save_path is None:
        fig.show()
    else:
        fig.savefig(save_path)


def dict_to_str(tree_dict):
    out_str = ['### Run Summary:\n']
    for k, v in tree_dict.items():

        if isinstance(v, dict):
            out = dict_to_str(v)
            out = out.replace('### Run Summary:\n',
                              f'- **{k}**')
            out_str.append(out.replace('\n', '\n\t'))

        else:
            out_str.append(f'-  **{k}**: {v}')

    return '\n'.join(out_str)