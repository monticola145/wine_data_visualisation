import statistics

import matplotlib.pyplot as plt
import pandas
import seaborn as sns

df = pandas.read_csv("winequality/winequality-red.csv", delimiter=";")
index_labels = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]


def binary_quality(quality=None):
    if quality >= 6:
        return 1
    return 0


def missing_data_heatmap(df=None):
    """Построение тепловой карты пропущенных значений"""
    colours = ["#ff0000", "#008000"]
    for index in index_labels:
        if "Пропуск" in list(df[index].fillna("Пропуск")):
            colours = ["#008000", "#ff0000"]
    sns.heatmap(df.isnull(), cmap=sns.color_palette(colours))
    plt.show()


def quartile_calculations(df=None):
    print("\n25% персентель:", list(df["quality"].quantile(q=[1 / 4]))[0])
    print("\n75% персентель:", list(df["quality"].quantile(q=[3 / 4]))[0])
    print(
        "\nМежквартильный диапазон:",
        1.5
        * (
            int(list(df["quality"].quantile(q=[3 / 4]))[0])
            - int(list(df["quality"].quantile(q=[1 / 4]))[0])
        ),
    )


def outliners_deleter(df=None, outliners=None):
    print(f"\nВыбросами являются значения: {outliners[0]} и {outliners[1]}")
    df = df[df.quality != int(outliners[0])]
    df = df[df.quality != int(outliners[1])]
    print("Выбросы успешно удалены")
    histogram_builder(
        data=df["quality"],
        title="График качества вин без выбросов",
        labelx="Качество по 10-балльной шкале",
        labely="Количество",
    )
    df["binary_quality"] = df["quality"].map(binary_quality)
    binary_quality_histogram_builder(
        data=df["binary_quality"],
        title="График качества вин без выбросов",
        labelx="Качество по 10-балльной шкале",
        labely="Количество",
    )
    mean_finder(df=df)
    boxplot_builder(df=df)
    auto_builder_visualisation(data=df)


def auto_builder_visualisation(data=None):
    for index in index_labels:
        if index != "quality":
            histogram_builder(
                data=df[index],
                title=f"График {index}",
                labelx=index,
                labely="Количество",
            )

    sns.heatmap(df.corr())
    plt.show()


def outliners_checker(df=None):
    sns.distplot(
        df["quality"],
        hist=True,
        kde=False,
        bins=int(180 / 5),
        color="red",
        hist_kws={"edgecolor": "black"},
    )
    plt.title("График качества вин с выбросами")
    plt.xlabel("Качество по 10-балльной шкале")
    plt.ylabel("Количество")
    plt.show()
    print(df["quality"].describe())
    outliners_deleter(
        df=df,
        outliners=[
            list(df["quality"].describe())[3],
            list(df["quality"].describe())[-1],
        ],
    )


def histogram_builder(data=None, title=None, labelx=None, labely=None):
    sns.distplot(
        data,
        hist=True,
        kde=False,
        bins=int(180 / 5),
        color="green",
        hist_kws={"edgecolor": "black"},
    )
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.show()


def binary_quality_histogram_builder(data=None, title=None, labelx=None, labely=None):
    sns.distplot(
        data,
        hist=True,
        kde=False,
        bins=int(180 / 10),
        color="green",
        hist_kws={"edgecolor": "black"},
    )
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.show()
    zeroes = list(data).count(0)
    non_zeroes = list(data).count(1)
    print(
        "Согласно графику, баланс бинарных классов составляет примерно:",
        zeroes / non_zeroes if zeroes > non_zeroes else non_zeroes / zeroes,
        "высокое качество к низкому"
        if zeroes > non_zeroes
        else "низкое качество к высокому",
    )


def boxplot_builder(df=None):
    df.boxplot(column="quality")
    plt.show()


def mean_finder(df=None):
    for index in index_labels:
        print(f"Медиана {index} =", round(statistics.mean(list(df[index])), 3))


def main():
    missing_data_heatmap(df=df)
    quartile_calculations(df=df)
    outliners_checker(df=df)


if __name__ == "__main__":
    main()
