import math
import re
import math
from typing import List, Tuple, Dict
from collections import Counter


def tokenize_text(text: str) -> List[str]:
    """Разбивает текст на слова, удаляя знаки пунктуации."""
    return re.findall(r"\b\w+\b", text.lower())


def calculate_tfidf(corpus):
    """Вычисляет TF-IDF для каждого слова в корпусе текстов."""
    tfidf = {}
    total_docs = len(corpus)
    term_frequencies = [Counter(doc.split()) for doc in corpus]

    for term in set(word for doc in corpus for word in doc.split()):
        doc_count = sum(1 for tf in term_frequencies if term in tf)
        idf = math.log((total_docs + 1) / (doc_count + 1)) + \
            1  # Добавили +1 для стабильности
        for doc_index, tf in enumerate(term_frequencies):
            tfidf[term] = tf[term] * idf

    return tfidf


def find_longest_common_subsequence(s1: str, s2: str) -> str:
    """Находит самую длинную общую подпоследовательность между двумя строками."""
    dp = [[""] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + s1[i - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)
    return dp[-1][-1]


def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    """Решает квадратное уравнение ax^2 + bx + c = 0."""
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("Нет действительных корней.")
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
    return root1, root2


def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """Решает задачу о рюкзаке методом динамического программирования."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]]
                )
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]


def matrix_multiplication(matrix1, matrix2):
    """Умножает две матрицы."""
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Incompatible matrix dimensions for multiplication")
    result = [[sum(a * b for a, b in zip(
        row,
        col
    )) for col in zip(*matrix2)] for row in matrix1]
    return result


def find_median_of_unsorted_array(arr: List[int]) -> float:
    """Находит медиану неотсортированного массива."""
    arr.sort()
    n = len(arr)
    mid = n // 2
    return arr[mid] if n % 2 else (arr[mid - 1] + arr[mid]) / 2


def detect_anagrams(word: str, candidates: List[str]) -> List[str]:
    """Находит все анаграммы слова среди списка кандидатов."""
    word_sorted = sorted(word)
    return [
        candidate
        for candidate in candidates
        if sorted(candidate) == word_sorted
    ]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Вычисляет расстояние Левенштейна между двумя строками."""
    dp = [
        [i + j if i * j == 0 else 0 for j in range(len(s2) + 1)]
        for i in range(len(s1) + 1)
    ]
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (s1[i - 1] != s2[j - 1]),
            )
    return dp[-1][-1]
