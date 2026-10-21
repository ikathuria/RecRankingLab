-- Share of each user's liked ratings (>= 4) that fall in each genre, ts <= $as_of only.
CREATE OR REPLACE TEMP TABLE user_genre AS
WITH liked AS (
    SELECT s.user_id, unnest(string_split(m.genres, '|')) AS genre
    FROM split_ratings s JOIN movies m USING (movie_id)
    WHERE s.ts <= $as_of AND s.rating >= 4.0
),
counts AS (SELECT user_id, genre, count(*) AS n FROM liked GROUP BY ALL)
SELECT user_id, genre, n::DOUBLE / sum(n) OVER (PARTITION BY user_id) AS share
FROM counts;
