import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Veri setlerini yükleyelim
results_df = pd.read_csv("./results.csv")

# Tarih sütununu datetime formatına çevirelim
results_df["date"] = pd.to_datetime(results_df["date"])

# Eksik değerleri kontrol edelim ve dolduralım
results_df.ffill(inplace=True)

# Model için gerekli olan özellikleri seçelim ve hazırlayalım
features = ["home_team", "away_team", "home_score", "away_score", "tournament", "date"]
results_df = results_df[features]

# Takım isimlerini sayısal değerlere dönüştürelim
teams = pd.concat([results_df["home_team"], results_df["away_team"]]).unique()
team_to_id = {team: idx for idx, team in enumerate(teams)}
results_df["home_team"] = results_df["home_team"].map(team_to_id)
results_df["away_team"] = results_df["away_team"].map(team_to_id)

# Hedef değişkeni oluşturalım: maç sonucu (0: Away Win, 1: Draw, 2: Home Win)
results_df["match_result"] = results_df.apply(
    lambda row: 2 if row["home_score"] > row["away_score"] else (1 if row["home_score"] == row["away_score"] else 0),
    axis=1
)

# Özellikler ve hedef değişkeni ayıralım
X = results_df[["home_team", "away_team"]]
y = results_df["match_result"]

# Veriyi normalize edelim
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Eğitim ve test setlerini bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN sınıflandırıcı modelini eğitelim
knn = KNeighborsClassifier()

# GridSearchCV ile parametre optimizasyonu yapalım
param_grid = {'n_neighbors': [5, 7, 9, 11, 13],
              'weights': ['uniform', 'distance'],
              'metric': ['manhattan', 'euclidean']}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi çapraz doğrulama doğruluğu:", grid_search.best_score_)

# En iyi modeli seçelim
best_knn = grid_search.best_estimator_

# Test seti üzerinde tahmin yapalım
y_pred = best_knn.predict(X_test)

# Doğruluk skorunu ve sınıflandırma raporunu yazdıralım
accuracy = accuracy_score(y_test, y_pred)
print(f"Test seti doğruluğu: {accuracy}")
print(classification_report(y_test, y_pred))

# Karışıklık matrisini yazdıralım
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Precision-Recall ve ROC eğrilerini çizdirelim
precision = dict()
recall = dict()
average_precision = dict()

# Binarize the output
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Compute Precision-Recall and ROC curve for each class
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], best_knn.predict_proba(X_test)[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='Class {}'.format(i))

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.title("Precision-Recall Curve")
plt.show()

# ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], best_knn.predict_proba(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Tahmin doğruluğunu görselleştirelim
correct_predictions = (y_pred == y_test)
plt.figure(figsize=(8, 6))
plt.bar(['Doğru', 'Yanlış'], pd.Series(correct_predictions).value_counts(), color=['green', 'red'])
plt.xlabel('Tahmin Doğruluğu')
plt.ylabel('Sayı')
plt.title('KNN Modeli Tahmin Doğruluğu')
plt.show()

# Takım isimlerini ve kimliklerini içeren bir sözlük oluşturalım
team_id_to_name = {idx: team for team, idx in team_to_id.items()}

# Tahmin yapmak için bir fonksiyon yazalım
def predict_match(home_team, away_team):
    home_team_id = team_to_id.get(home_team)
    away_team_id = team_to_id.get(away_team)
    if home_team_id is None:
        raise ValueError(f"Geçersiz takım ismi: {home_team}")
    if away_team_id is None:
        raise ValueError(f"Geçersiz takım ismi: {away_team}")
    
    prediction = best_knn.predict([[home_team_id, away_team_id]])[0]
    return prediction

# 2024 UEFA EURO gruplarındaki maçların sonuçlarını tahmin edelim
groups = {
    "Group A": ["Germany", "Hungary", "Switzerland", "Scotland"],
    "Group B": ["Spain", "Croatia", "Italy", "Albania"],
    "Group C": ["Slovenia", "Denmark", "Serbia", "England"],
    "Group D": ["Poland", "Netherlands", "Austria", "France"],
    "Group E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
    "Group F": ["Turkey", "Georgia", "Portugal", "Czech Republic"]
}

# Gruplardaki tüm maçları tahmin edelim
group_results = {}
for group, teams in groups.items():
    matches = [(teams[i], teams[j]) for i in range(len(teams)) for j in range(i+1, len(teams))]
    group_results[group] = []
    for match in matches:
        home_team, away_team = match
        result = predict_match(home_team, away_team)
        if result == 2:
            match_result = "Home Win"
        elif result == 1:
            match_result = "Draw"
        else:
            match_result = "Away Win"
        group_results[group].append((home_team, away_team, match_result))

# Gruplardan çıkan takımları belirleme
group_standings = {}
for group, results in group_results.items():
    standings = {team: 0 for team in groups[group]}
    for result in results:
        home_team, away_team, match_result = result
        if match_result == "Home Win":
            standings[home_team] += 3
        elif match_result == "Draw":
            standings[home_team] += 1
            standings[away_team] += 1
        elif match_result == "Away Win":
            standings[away_team] += 3
    sorted_standings = sorted(standings.items(), key=lambda item: item[1], reverse=True)
    group_standings[group] = sorted_standings

# Üçüncü sıradaki takımları belirle ve üçüncülük maçlarını oluştur
third_place_teams = []
for group, standings in group_standings.items():
    all_standings = sorted(standings, key=lambda item: item[1], reverse=True)
    if len(all_standings) >= 3:
        third_place_teams.append((all_standings[2][0], group))

# Üçüncüler grubundaki maçları tahmin et
third_place_results = []
third_place_standings = {team: 0 for team, group in third_place_teams}
for i in range(len(third_place_teams)):
    for j in range(i + 1, len(third_place_teams)):
        home_team = third_place_teams[i][0]
        away_team = third_place_teams[j][0]
        result = predict_match(home_team, away_team)
        if result == 2:
            match_result = "Home Win"
            third_place_standings[home_team] += 3
        elif result == 1:
            match_result = "Draw"
            third_place_standings[home_team] += 1
            third_place_standings[away_team] += 1
        else:
            match_result = "Away Win"
            third_place_standings[away_team] += 3
        third_place_results.append((home_team, away_team, match_result))

# Üçüncüler grubundan çıkan ilk 4 takımı belirle
sorted_third_place_standings = sorted(third_place_standings.items(), key=lambda item: item[1], reverse=True)
top_four_third_place_teams = [team for team, points in sorted_third_place_standings[:4]]

# 16'lık grupta maçları belirle
round_of_16_matches = [
    (group_standings["Group B"][0][0], top_four_third_place_teams[0]),
    (group_standings["Group A"][0][0], group_standings["Group C"][1][0]),
    (group_standings["Group F"][0][0], top_four_third_place_teams[1]),
    (group_standings["Group D"][1][0], group_standings["Group E"][1][0]),
    (group_standings["Group E"][0][0], top_four_third_place_teams[2]),
    (group_standings["Group D"][0][0], group_standings["Group F"][1][0]),
    (group_standings["Group C"][0][0], top_four_third_place_teams[3]),
    (group_standings["Group A"][1][0], group_standings["Group B"][1][0])
]

# 16'lık grup maç sonuçlarını tahmin et
round_of_16_results = []
for match in round_of_16_matches:
    home_team, away_team = match
    result = predict_match(home_team, away_team)
    if result == 2:
        match_result = "Home Win"
    elif result == 1:
        match_result = "Draw"
    else:
        match_result = "Away Win"
    round_of_16_results.append((home_team, away_team, match_result))

# Çeyrek final eşleşmeleri
quarter_finals = [
    (round_of_16_results[0][0], round_of_16_results[1][0]),
    (round_of_16_results[2][0], round_of_16_results[3][0]),
    (round_of_16_results[4][0], round_of_16_results[5][0]),
    (round_of_16_results[6][0], round_of_16_results[7][0])
]

# Çeyrek final maç sonuçlarını tahmin et
quarter_final_results = []
for match in quarter_finals:
    home_team, away_team = match
    result = predict_match(home_team, away_team)
    if result == 2:
        match_result = "Home Win"
    elif result == 1:
        match_result = "Draw"
    else:
        match_result = "Away Win"
    quarter_final_results.append((home_team, away_team, match_result))

# Yarı final eşleşmeleri
semi_finals = [
    (quarter_final_results[0][0], quarter_final_results[1][0]),
    (quarter_final_results[2][0], quarter_final_results[3][0])
]

# Yarı final maç sonuçlarını tahmin et
semi_final_results = []
for match in semi_finals:
    home_team, away_team = match
    result = predict_match(home_team, away_team)
    if result == 2:
        match_result = "Home Win"
    elif result == 1:
        match_result = "Draw"
    else:
        match_result = "Away Win"
    semi_final_results.append((home_team, away_team, match_result))

# Final eşleşmesi
finals = [
    (semi_final_results[0][0], semi_final_results[1][0])
]

# Final maç sonucunu tahmin et
final_results = []
for match in finals:
    home_team, away_team = match
    result = predict_match(home_team, away_team)
    if result == 2:
        match_result = "Home Win"
    elif result == 1:
        match_result = "Draw"
    else:
        match_result = "Away Win"
    final_results.append((home_team, away_team, match_result))
    
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.table import Table

def draw_group_stage_results_and_standings(group_results, group_standings):
    num_groups = len(group_results)
    fig, ax = plt.subplots(num_groups, 1, figsize=(8, 6 * num_groups))

    if num_groups == 1:
        ax = [ax]

    for idx, group in enumerate(group_results.keys()):
        # Extract match results and team standings
        match_data = [(f"{home_team} vs {away_team}", match_result) for home_team, away_team, match_result in group_results[group]]
        team_data = [(team, points) for team, points in group_standings[group]]

        ax[idx].axis('off')
        ax[idx].set_title(f'Group {group} Results and Standings', fontsize=12, weight='bold')

        # Create table for match results
        table_matches = Table(ax[idx], bbox=[0, 0.6, 1, 0.4])
        table_matches.auto_set_font_size(False)
        table_matches.set_fontsize(8)

        # Add headers
        table_matches.add_cell(0, 0, 1, 1, text='Match', loc='center', edgecolor='black')
        table_matches.add_cell(0, 1, 1, 1, text='Result', loc='center', edgecolor='black')

        for i, (match, result) in enumerate(match_data, start=1):
            table_matches.add_cell(i, 0, 1, 1, text=match, loc='center', edgecolor='black')
            table_matches.add_cell(i, 1, 1, 1, text=result, loc='center', edgecolor='black')

        for key, cell in table_matches.get_celld().items():
            cell.set_linewidth(1)
            if key[0] == 0:
                cell.set_text_props(weight='bold')

        ax[idx].add_table(table_matches)

        # Create table for team standings
        table_standings = Table(ax[idx], bbox=[0, 0, 1, 0.4])
        table_standings.auto_set_font_size(False)
        table_standings.set_fontsize(8)

        # Add headers
        table_standings.add_cell(0, 0, 1, 1, text='Team', loc='center', edgecolor='black')
        table_standings.add_cell(0, 1, 1, 1, text='Points', loc='center', edgecolor='black')

        for i, (team, points) in enumerate(team_data, start=1):
            table_standings.add_cell(i, 0, 1, 1, text=team, loc='center', edgecolor='black')
            table_standings.add_cell(i, 1, 1, 1, text=points, loc='center', edgecolor='black')

        for key, cell in table_standings.get_celld().items():
            cell.set_linewidth(1)
            if key[0] == 0:
                cell.set_text_props(weight='bold')

        ax[idx].add_table(table_standings)

    plt.tight_layout()
    plt.show()


def draw_elbow_bracket(round_of_16, quarter_finals, semi_finals, final):
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('white')
    plt.axis('off')

    # Define positions for the matches
    y_positions_r16 = list(range(15, -1, -2))  # Y positions for the round of 16 matches
    y_positions_qf = [(y_positions_r16[i] + y_positions_r16[i+1]) / 2 for i in range(0, len(y_positions_r16), 2)]
    y_positions_sf = [(y_positions_qf[i] + y_positions_qf[i+1]) / 2 for i in range(0, len(y_positions_qf), 2)]
    y_position_final = [(y_positions_sf[0] + y_positions_sf[1]) / 2]

    # Draw matches and connect with lines
    def draw_team_box(x, y, team):
        box_width = 2
        box_height = 0.8
        ax.add_patch(patches.Rectangle((x, y - box_height / 2), box_width, box_height, edgecolor='black', facecolor='lightblue', linewidth=1))
        ax.text(x + box_width / 2, y, team, ha='center', va='center', fontsize=8)

    def draw_elbow(x1, y1, x2, y2, winner):
        mid_x1 = (x1 + x2) / 2
        mid_y1 = (y1 + y2) / 2
        mid_y2 = (y1 + y2) / 2
        ax.plot([x1, mid_x1, mid_x1, x2], [y1, y1, y2, y2], color='black', lw=1)
        ax.text(mid_x1, mid_y1, winner, ha='center', va='bottom', fontsize=8, color='green')

    # Plot matches
    for i, match in enumerate(round_of_16):
        home_team, away_team, result = match
        draw_team_box(0, y_positions_r16[i], home_team)
        draw_team_box(0, y_positions_r16[i] - 1, away_team)
        draw_elbow(2, y_positions_r16[i] - 0.5, 3, y_positions_qf[i // 2], home_team if "Home" in result else away_team)

    for i, match in enumerate(quarter_finals):
        home_team, away_team, result = match
        draw_team_box(3, y_positions_qf[i], home_team)
        draw_team_box(3, y_positions_qf[i] - 1, away_team)
        draw_elbow(5, y_positions_qf[i] - 0.5, 6, y_positions_sf[i // 2], home_team if "Home" in result else away_team)

    for i, match in enumerate(semi_finals):
        home_team, away_team, result = match
        draw_team_box(6, y_positions_sf[i], home_team)
        draw_team_box(6, y_positions_sf[i] - 1, away_team)
        draw_elbow(8, y_positions_sf[i] - 0.5, 9, y_position_final[0], home_team if "Home" in result else away_team)

    final_home_team, final_away_team, final_result = final[0]
    draw_team_box(9, y_position_final[0], final_home_team)
    draw_team_box(9, y_position_final[0] - 1, final_away_team)
    ax.text(12,y_position_final[0], "Winner is" ,ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(12, y_position_final[0] - 0.5, final_home_team if "Home" in final_result else final_away_team, ha='center', va='center', fontsize=12, fontweight='bold')

    plt.show()

# Example usage:
draw_group_stage_results_and_standings(group_results, group_standings)
draw_elbow_bracket(round_of_16_results, quarter_final_results, semi_final_results, final_results)