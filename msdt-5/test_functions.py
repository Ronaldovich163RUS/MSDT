import pytest

from functions import (tokenize_text, calculate_tfidf, find_longest_common_subsequence,
                       solve_quadratic, knapsack, matrix_multiplication,
                       find_median_of_unsorted_array, detect_anagrams,
                       levenshtein_distance)


@pytest.mark.parametrize("text, expected", [
    ("hello world", ["hello", "world"]),
    ("Testing functions.", ["testing", "functions"]),
    ("Козлов Анатолий Романович", ["козлов", "анатолий", "романович"])
])
def test_tokenize_text(text, expected):
    assert tokenize_text(text) == expected


def test_calculate_tfidf():
    corpus = ["this is a test", "this is another test"]
    tfidf = calculate_tfidf(corpus)
    assert "this" in tfidf
    assert tfidf["test"] > 0
    assert tfidf["another"] > tfidf["this"]


@pytest.mark.parametrize("s1, s2, expected", [
    ("abcde", "ace", "ace"),
    ("abc", "abc", "abc"),
    ("abc", "def", "")
])
def test_find_longest_common_subsequence(s1, s2, expected):
    assert find_longest_common_subsequence(s1, s2) == expected


@pytest.mark.parametrize("a, b, c, expected", [
    (1, -3, 2, (2, 1)),
    (1, -2, 1, (1, 1)),
])
def test_solve_quadratic(a, b, c, expected):
    assert solve_quadratic(a, b, c) == pytest.approx(expected)


def test_solve_quadratic_no_real_roots():
    with pytest.raises(ValueError):
        solve_quadratic(1, 0, 1)


def test_knapsack():
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    assert knapsack(weights, values, capacity) == 7


def test_matrix_multiplication():
    matrix1 = [[1, 2], [3, 4]]
    matrix2 = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    assert matrix_multiplication(matrix1, matrix2) == expected


def test_matrix_multiplication_incompatible():
    matrix1 = [[1, 2, 3]]
    matrix2 = [4, 5]
    with pytest.raises(ValueError):
        matrix_multiplication(matrix1, matrix2)


@pytest.mark.parametrize("arr, expected", [
    ([3, 1, 2], 2),
    ([3, 1, 2, 4], 2.5)
])
def test_find_median_of_unsorted_array(arr, expected):
    assert find_median_of_unsorted_array(arr) == expected


@pytest.mark.parametrize("word, candidates, expected", [
    ("listen", ["enlist", "google", "inlets", "banana"], ["enlist", "inlets"]),
    ("evil", ["vile", "veil", "live"], ["vile", "veil", "live"])
])
def test_detect_anagrams(word, candidates, expected):
    assert detect_anagrams(word, candidates) == expected


@pytest.mark.parametrize("s1, s2, expected", [
    ("kitten", "sitting", 3),
    ("flaw", "lawn", 2),
    ("intention", "execution", 5)
])
def test_levenshtein_distance(s1, s2, expected):
    assert levenshtein_distance(s1, s2) == expected
