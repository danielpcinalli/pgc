import matplotlib.pyplot as plt
import inspect
import datasets as ds
import seaborn as sns
import numpy as np


def get_load_functions():

    all_loaders = []
    for name, function in inspect.getmembers(ds, predicate=inspect.isfunction):
        if  name[:4] == 'load':
            all_loaders.append(function)
    return all_loaders


def main():
    all_loaders = get_load_functions()
    func_names = [func.__name__[5:] for func in all_loaders]


    for load_func, func_name in zip(all_loaders, func_names):

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(4, 4)

        plt.subplots_adjust(bottom=.1)
        #info on shapes
        ax1 = fig.add_subplot(gs[0, 0])
        df = load_func(format='pd')

        #df.iloc[:, :-1] = ds.z_score_normalize(df.iloc[:, :-1])

        instances = len(df)
        amount_classes = len(df['class'].unique())
        amount_attributes = len(df.columns) - 1
        amount_nans = df.isna().sum().sum()
        duplicated_rows_mask = df.duplicated(keep='first')
        duplicated_rows = len(df[duplicated_rows_mask])
        df = df.drop_duplicates()
        instances_after_cleanup = len(df)

        text = ''.join([
            f'{instances} instâncias, com {duplicated_rows} linha(s) duplicadas\n',
            f'{instances_after_cleanup} instâncias após limpeza\n',
            f'{amount_attributes} atributos\n',
            f'{amount_classes} classes\n',
            f'{amount_nans} valores NaN\n',
            ])

        plt.text(0, 0, text)
        ax1.axis('off')

        #contagem de nans
        # ax2 = fig.add_subplot(gs[3, :])
        # nans_per_class = df.isna().sum()
        # nans_per_class.plot(kind='bar', sort_columns=True)

        # for p in ax2.patches:
        #     x = p.get_x() + p.get_width() / 2
        #     y = p.get_y() + p.get_height()
        #     heigth = p.get_height()
        #     ax2.annotate(heigth, (x, y), ha='center', va='bottom')


        # plt.ylim(0, None)
        # plt.xticks(rotation=60)
        # plt.title('Contagem de NANs')



        #contagem de instâncias por classe
        ax3 = fig.add_subplot(gs[1, :])
        sns.countplot(data=df, x='class', order=df['class'].value_counts().index)

        for p in ax3.patches:
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + p.get_height()
            heigth = p.get_height()
            t_color = 'black'
            if y > .8 * ax3.get_ylim()[1]:
                y = .85 * y
                t_color = 'white'
            ax3.annotate(heigth, (x, y), ha='center', va='bottom', color=t_color)

        plt.xticks(rotation=30)
        plt.title('Contagem de classes')
        plt.xlabel("Classes")
        plt.ylabel("Contagem")


        #tipos
        # ax4 = fig.add_subplot(gs[2, 0])
        # plt.text(0, 0, str(df.dtypes))
        # ax4.axis('off')

        #violinplot
        # ax5 = fig.add_subplot(gs[1:, :])
        # sns.boxplot(data = df)
        # plt.xticks(rotation=45)

        #contagem de valores zero
        ax6 = fig.add_subplot(gs[3:, :])
        zero_count = df.iloc[:, :-1][df.iloc[:, :-1] == 0].count()
        zero_count.plot(kind='bar')

        plt.title("Contagem de valores zero")
        plt.xlabel("Atributos")
        plt.ylabel("Contagem")
        plt.xticks(rotation=45)
        plt.ylim(0, None)

        plt.suptitle(func_name)
        #plt.show()
        plt.savefig(f'./info_datasets_output/{func_name}.png')

if __name__=='__main__':
    main()
