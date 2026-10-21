-- One row per user, from ratings with ts <= $as_of only. last_ts is kept so tests can assert it.
CREATE OR REPLACE TEMP TABLE user_features AS
SELECT user_id,
       count(*)                           AS u_n_ratings,
       avg(rating)                        AS u_mean_rating,
       avg((rating >= 4.0)::DOUBLE)       AS u_pos_rate,
       ($as_of - max(ts)) / 86400.0       AS u_days_since_last,
       (max(ts) - min(ts)) / 86400.0      AS u_span_days,
       max(ts)                            AS last_ts
FROM split_ratings
WHERE ts <= $as_of
GROUP BY user_id;
