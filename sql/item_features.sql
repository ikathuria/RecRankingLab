-- One row per item, from ratings with ts <= $as_of only. Release year is parsed from the title.
CREATE OR REPLACE TEMP TABLE item_features AS
WITH r AS (
    SELECT movie_id,
           count(*)                                              AS i_n_ratings,
           sum((rating >= 4.0)::INT)                             AS i_n_pos,
           avg(rating)                                           AS i_mean_rating,
           avg((rating >= 4.0)::DOUBLE)                          AS i_pos_rate,
           sum((rating >= 4.0 AND ts > $as_of - 90 * 86400)::INT) AS i_pos_90d,
           max(ts)                                               AS last_ts
    FROM split_ratings
    WHERE ts <= $as_of
    GROUP BY movie_id
)
SELECT r.*,
       year(to_timestamp($as_of)) - TRY_CAST(regexp_extract(m.title, '\((\d{4})\)\s*$', 1) AS INT)
           AS i_age_years,
       len(string_split(m.genres, '|'))                          AS i_n_genres
FROM r JOIN movies m USING (movie_id);
