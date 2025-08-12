### Fonctions utiles pour l'échantillonnage"
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import random

from collections import Counter
from collections import defaultdict


def verif_coherence(df, enjeu):
    # Colonnes liées à l'enjeu
    autres_cols = [col for col in df.columns if col != enjeu]

    # 1. Vérifier que si climat est NaN, toutes les autres le sont aussi
    condition1 = df[df[enjeu].isna()][autres_cols].isna().all(axis=1).all()

    # 2. Vérifier que si climat n'est PAS NaN, alors au moins une autre colonne a une valeur
    condition2 = df[~df[enjeu].isna()][autres_cols].notna().any(axis=1).all()

    # Résultat global
    print(f"Condition 1 ({enjeu} NaN ⇒ tout le reste aussi NaN) :", condition1)
    print(f"Condition 2 ({enjeu} non NaN ⇒ au moins une autre valeur) :", condition2)

    if condition1 and condition2:
        print("✅ Données cohérentes.")
    elif condition2:
        # Condition 1 : si enjeu est NaN, alors tout le reste doit l’être aussi
        mask_enjeu_na = df[enjeu].isna()
        incoh1 = df.loc[mask_enjeu_na, autres_cols].notna().any(axis=1)
        if incoh1.any():
            print(
                f"❌ Incohérences sur condition 1 ({enjeu} NaN mais données présentes ailleurs) :"
            )
            print(df.loc[mask_enjeu_na].loc[incoh1])
    else:
        print("❌ Incohérences détectées.")


def selection_articles_enjeu(
    df,
    enjeux_dict,
    chaine_causale_dict,
    nb_art_enjeu,
    ratio_nat_reg=0.33,
    dossier_nat="articles_pdf_national",
    dossier_reg="articles_pdf_regional",
):
    tirages_par_enjeu = []

    # Tirer "nb_art_enjeu" lignes par enjeu, réparties dans les catégories causales et selon le ratio national/régional
    for enjeu, colonnes in chaine_causale_dict.items():
        enjeu_col = enjeux_dict[enjeu]
        df_enjeu = df[df[enjeu_col] == 1].copy()

        nb_types = len(colonnes)
        cartes_par_type = nb_art_enjeu // nb_types  # ex: 12 // 4 = 3
        n_nat = int(np.round(cartes_par_type * ratio_nat_reg))
        n_reg = cartes_par_type - n_nat

        for col in colonnes:
            subset = df_enjeu[df_enjeu[col] > 0].copy()

            # Séparer national et régional
            subset_nat = subset[subset["Dossier"] == dossier_nat]
            subset_reg = subset[subset["Dossier"] == dossier_reg]

            tirage_nat = subset_nat.sample(
                n=min(n_nat, len(subset_nat)), random_state=42
            )
            tirage_reg = subset_reg.sample(
                n=min(n_reg, len(subset_reg)), random_state=42
            )

            # Fusionner les deux tirages
            tirage = pd.concat([tirage_nat, tirage_reg], ignore_index=True)

            # Compléter s'il manque des lignes (ex. si échantillons trop petits)
            if len(tirage) < cartes_par_type:
                reste = subset.drop(tirage.index, errors="ignore")
                supplement = reste.sample(
                    n=cartes_par_type - len(tirage), random_state=42
                )
                tirage = pd.concat([tirage, supplement], ignore_index=True)

            tirage = tirage.copy()
            tirage["enjeu"] = enjeu
            tirage["type"] = col
            tirages_par_enjeu.append(tirage)

    # Fusion finale
    tirages_enjeux = pd.concat(tirages_par_enjeu, ignore_index=True)

    # ✅ Affichage
    print(f"Nombre total de lignes tirées : {len(tirages_enjeux)}")
    print("\nRépartition par Dossier :")
    print(tirages_enjeux["Dossier"].value_counts())

    # Output
    return tirages_enjeux


def selection_articles_sans_enjeu(df, keyword_thresh, nb_art_enjeu):
    tirages_sans_enjeu = []

    # Définir le nombre de types de presse présents
    n_type_presse = df["Dossier"].nunique()

    # Nombre de types dans les "sans enjeu"
    types_sans_enjeu = ["0 mots-clés", f"<{keyword_thresh} mots-clés"]
    cartes_par_type = nb_art_enjeu // len(types_sans_enjeu)  # 6 si nb_art_enjeu=12
    cartes_par_dossier = cartes_par_type // n_type_presse

    # a. Zéro mot-clé sur tous les enjeux
    zero_keywords = df[
        (df["climat_not_HRFP_total_keywords"] == 0)
        & (df["biodiversite_not_HRFP_total_keywords"] == 0)
        & (df["ressources_not_HRFP_total_keywords"] == 0)
    ].copy()

    # b. 1 à <keyword_thresh mots-clés sur au moins un enjeu
    low_keywords = df[
        (
            (df["climat_not_HRFP_total_keywords"] > 0)
            & (df["climat_not_HRFP_total_keywords"] < keyword_thresh)
        )
        | (
            (df["biodiversite_not_HRFP_total_keywords"] > 0)
            & (df["biodiversite_not_HRFP_total_keywords"] < keyword_thresh)
        )
        | (
            (df["ressources_not_HRFP_total_keywords"] > 0)
            & (df["ressources_not_HRFP_total_keywords"] < keyword_thresh)
        )
    ].copy()

    # c. Fonction de tirage équilibré par Dossier
    def tirage_sans_enjeu(df_source, cartes_par_type, cartes_par_dossier, type_label):
        # S'assurer que chaque Dossier a suffisamment de lignes
        group_sizes = df_source.groupby("Dossier").size()
        eligible_dossiers = group_sizes[group_sizes >= cartes_par_dossier].index
        subset = df_source[df_source["Dossier"].isin(eligible_dossiers)]

        if len(subset) >= cartes_par_type:
            tirage = (
                subset.groupby("Dossier", group_keys=False)
                .apply(lambda x: x.sample(n=cartes_par_dossier, random_state=42))
                .reset_index(drop=True)
                .head(cartes_par_type)
            )
        else:
            tirage = df_source.sample(n=cartes_par_type, random_state=42)

        tirage["enjeu"] = "aucun"
        tirage["type"] = type_label
        return tirage

    # d. Tirages
    tirages_sans_enjeu.append(
        tirage_sans_enjeu(
            zero_keywords, cartes_par_type, cartes_par_dossier, "0 mots-clés"
        )
    )
    tirages_sans_enjeu.append(
        tirage_sans_enjeu(
            low_keywords,
            cartes_par_type,
            cartes_par_dossier,
            f"<{keyword_thresh} mots-clés",
        )
    )

    # e. Fusion
    tirages_sans_enjeu = pd.concat(tirages_sans_enjeu, ignore_index=True)
    print(f"Nombre total de lignes 'sans enjeu' tirées : {len(tirages_sans_enjeu)}")


def visualize_selection(selection, plot="word_counts"):

    if plot == "word_counts":
        # Visualisation de la longueur des articles
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

        # Liste des enjeux à tracer
        enjeux_list = ["climat", "biodiversité", "ressources", "aucun"]

        # Palette et types
        types = selection["type"].unique()
        palette = dict(zip(types, sns.color_palette("Set1", len(types))))

        for i, enjeu in enumerate(enjeux_list):
            ax = axes[i]
            subset = selection[selection["enjeu"] == enjeu]

            sns.swarmplot(
                data=subset,
                x="type",
                y="word_count",
                hue="type",
                ax=ax,
                palette=palette,
            )

            ax.set_title(enjeu.capitalize())
            ax.set_xlabel("")
            ax.set_ylabel("Longueur de l'article (Nombre de mots)" if i == 0 else "")
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

            # Légende manuelle sous chaque plot
            from matplotlib.patches import Patch

            handles = [
                Patch(color=palette[t], label=t) for t in subset["type"].unique()
            ]
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=1,
                title="Thématique",
                frameon=False,
            )

        fig.suptitle("Exploration de la sélection d'articles", fontsize=16)
        plt.tight_layout()
        plt.show()

    elif plot == "source_type":
        # Représentativité des types
        order = selection["type"].value_counts().index

        # Tracer le graphique
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.countplot(selection, x="Dossier", ax=ax[0])
        sns.countplot(
            data=selection,
            y="type",
            hue="Dossier",
            order=order,
            palette="rainbow",
            ax=ax[1],
        )

        plt.title("Nombre d'articles")
        ax[1].legend(
            loc="lower left",
            bbox_to_anchor=(-0.5, -0.5),
            ncol=1,
            title="Périodicité",
            frameon=False,
        )
        plt.tight_layout()
        plt.show()

    elif plot == "source_name":
        # Vérification de l'équilibre des sources
        # Table de contingence : nombre d'articles par source et type
        pivot = selection.pivot_table(
            index="source_name", columns="type", aggfunc="size", fill_value=0
        )

        # Réordonner les sources selon leur fréquence totale
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

        # Graphique empilé
        pivot.plot(kind="barh", stacked=True, figsize=(10, 8), colormap="tab20b")
        plt.title("Nombre d'articles par rédaction")
        plt.xlabel("Nombre d'articles")
        plt.ylabel("Source")
        plt.legend(
            title="Type", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False
        )
        plt.tight_layout()
        plt.show()

    else:
        raise (
            f"plot should be one of 'word_counts', 'source_type' or source_name'. You provided {plot}"
        )


def attribute_articles_veilleurs(
    selection, n_article_par_veilleur=10, lecteurs_par_article=5
):
    # Nombre total d'articles
    n_articles = selection.shape[0]

    # Nombre total de veilleurs nécessaires
    n_veilleurs = math.ceil(n_articles / n_article_par_veilleur)

    # Mélanger les articles
    selection_shuffled = selection.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    # Générer les IDs de veilleurs pour tous les articles
    veilleur_ids = [
        f"veilleur_{i+1}"
        for i in range(n_veilleurs)
        for _ in range(n_article_par_veilleur)
    ]

    # Couper la liste à la bonne taille (exactement le nombre d'articles)
    veilleur_ids = veilleur_ids[:n_articles]

    # Assigner les veilleurs aux articles
    selection_shuffled["veilleur_id"] = veilleur_ids

    # Paramètres
    n_articles = len(selection)

    # Calcul du nombre de veilleurs nécessaires
    n_veilleurs = (
        math.ceil(n_articles * lecteurs_par_article / n_article_par_veilleur) + 1
    )  # On rajoute un veilleur par sécurité
    print(
        f"Avec maximum {n_article_par_veilleur} articles par veilleurs, il faut {n_veilleurs} veilleurs."
    )

    veilleur_ids = [f"veilleur_{i+1}" for i in range(n_veilleurs)]
    random.shuffle(veilleur_ids)  # pour varier les ordres

    # Initialiser les affectations
    veilleur_articles = defaultdict(set)
    article_veilleurs = defaultdict(set)
    attributions = []

    # Suivi de la charge restante
    charge_restante = {v: n_article_par_veilleur for v in veilleur_ids}

    # Pour chaque article
    for idx, article in selection.iterrows():
        # Trouver les veilleurs avec charge disponible
        candidats = [
            v
            for v in veilleur_ids
            if charge_restante[v] > 0 and idx not in veilleur_articles[v]
        ]
        if len(candidats) < lecteurs_par_article:
            raise ValueError(
                f"Article {idx} : pas assez de veilleurs disponibles ({len(candidats)} candidats)."
            )

        choisis = random.sample(candidats, lecteurs_par_article)

        for v in choisis:
            veilleur_articles[v].add(idx)
            charge_restante[v] -= 1
            attributions.append(
                {"article_id": idx, "veilleur_id": v, **article.to_dict()}
            )

    # Finaliser
    attributions_df = pd.DataFrame(attributions)
    print("✅ Attribution réussie.")

    return attributions_df


def check_attribution(attributions_df, print_stats=False, plot="nb_articles"):
    if plot == "nb_articles":
        # Verification que chaque veilleur a le bon nombre d'article
        attributions_df["veilleur_id"].value_counts().plot(
            kind="barh", title="Nombre d'articles par veilleur"
        )

    elif plot == "boxplot_veilleurs":
        sns.boxplot(attributions_df, y="veilleur_id", x="word_count")
        plt.title("Longueur moyenne des articles par veilleurs")
    else:
        KeyError(
            f"plot should be one of 'nb_article' or 'boxplot_veilleurs'. You provided {plot}"
        )

    # Verification de la répartition des articles par veilleurs, en termes de longueur d'articles
    # Statistiques descriptives par veilleur sur la longueur des articles
    if print_stats:
        stats_par_veilleur = (
            attributions_df.groupby("veilleur_id")["word_count"]
            .describe()[["count", "mean", "min", "max", "std"]]
            .sort_values("mean")
            .reset_index()
        )
        print(stats_par_veilleur)
