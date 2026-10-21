-- Features for each (user_id, movie_id) in temp table `cand` (also carries knn_score and the
-- source ranks). genre_match = mean over the item's genres of the user's liked-genre share.
CREATE OR REPLACE TEMP TABLE pair_features AS
WITH item_genre AS (
    SELECT movie_id, unnest(string_split(genres, '|')) AS genre FROM movies
),
gm AS (
    SELECT c.user_id, c.movie_id, avg(coalesce(ug.share, 0.0)) AS genre_match
    FROM cand c
    JOIN item_genre ig ON ig.movie_id = c.movie_id
    LEFT JOIN user_genre ug ON ug.user_id = c.user_id AND ug.genre = ig.genre
    GROUP BY ALL
)
SELECT c.user_id, c.movie_id, c.knn_score, c.pop_rank, c.recent_rank, c.knn_rank,
       coalesce(gm.genre_match, 0.0)             AS genre_match,
       i.i_n_ratings, i.i_n_pos, i.i_mean_rating, i.i_pos_rate, i.i_pos_90d, i.i_age_years,
       i.i_n_genres,
       u.u_n_ratings, u.u_mean_rating, u.u_pos_rate, u.u_days_since_last, u.u_span_days,
       i.i_mean_rating - u.u_mean_rating         AS item_vs_user_mean
FROM cand c
JOIN item_features i USING (movie_id)
JOIN user_features u USING (user_id)
LEFT JOIN gm USING (user_id, movie_id);
