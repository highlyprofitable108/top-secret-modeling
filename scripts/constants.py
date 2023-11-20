# constants.py

# List of column names to keep
COLUMNS_TO_KEEP = [
    "odds.spread_close",
    "passing.totals.hurries_difference",
    "passing.totals.knockdowns_difference",
    "passing.totals.redzone_attempts_difference",
    "rushing.totals.attempts_difference",
    "rushing.totals.avg_yards_ratio",
    "rushing.totals.broken_tackles_difference",
    "rushing.totals.yards_difference",
    "receiving.totals.air_yards_difference",
    "receiving.totals.dropped_passes_difference",
    "receiving.totals.touchdowns_difference",
    "receiving.totals.yards_difference",
    "receiving.totals.yards_after_contact_difference",
    "summary.penalties_difference",
    "summary.possession_time_difference",
    "summary.return_yards_difference",
    "summary.rush_plays_difference",
    "summary.total_yards_difference",
    "efficiency.redzone.successes_difference",
    "efficiency.thirddown.pct_ratio",
    "defense.totals.blitzes_difference",
    "defense.totals.combined_difference",
    "defense.totals.forced_fumbles_difference",
    "defense.totals.fumble_recoveries_difference",
    "defense.totals.hurries_difference",
    "defense.totals.interceptions_difference",
    "defense.totals.misc_fumble_recoveries_difference",
    "defense.totals.misc_tackles_difference",
    "defense.totals.passes_defended_difference",
    "defense.totals.sacks_difference",
    "defense.totals.safeties_difference",
    "defense.totals.sp_assists_difference",
    "defense.totals.sp_forced_fumbles_difference",
    "defense.totals.sp_tackles_difference",
    "defense.totals.tloss_difference",
    "advanced.drop_rate_difference",
    "advanced.pressure_rate_on_qb_difference",
    "advanced.tackle_for_loss_rate_difference",
    "extra_points.conversions.totals.pass_attempts_difference",
    "extra_points.conversions.totals.rush_attempts_difference",
    "extra_points.kicks.totals.attempts_difference",
    "extra_points.kicks.totals.blocked_difference",
    "extra_points.kicks.totals.pct_difference",
    "field_goals.totals.pct_ratio",
    "first_downs.pass_difference",
    "first_downs.penalty_difference",
    "first_downs.total_difference",
    "fumbles.totals.ez_rec_tds_difference",
    "fumbles.totals.forced_fumbles_difference",
    "fumbles.totals.opp_rec_tds_difference",
    "fumbles.totals.opp_rec_yards_difference",
    "fumbles.totals.own_rec_difference",
    "fumbles.totals.own_rec_tds_difference",
    "int_returns.totals.avg_yards_difference",
    "int_returns.totals.touchdowns_difference",
    "interceptions.number_difference",
    "kick_returns.totals.number_difference",
    "kickoffs.totals.inside_20_difference",
    "kickoffs.totals.return_yards_difference",
    "kickoffs.totals.touchbacks_difference",
    "kickoffs.totals.yards_difference",
    "misc_returns.totals.blk_fg_touchdowns_difference",
    "misc_returns.totals.ez_rec_touchdowns_difference",
    "misc_returns.totals.yards_difference",
    "penalties.totals.penalties_difference",
    "punt_returns.totals.faircatches_difference",
    "punt_returns.totals.longest_difference",
    "punt_returns.totals.number_difference",
    "punts.totals.attempts_difference",
    "punts.totals.avg_hang_time_ratio",
    "punts.totals.avg_net_yards_ratio",
    "punts.totals.blocked_difference",
    "punts.totals.inside_20_difference",
    "punts.totals.longest_difference",
    "touchdowns.fumble_return_difference",
    "touchdowns.int_return_difference",
    "touchdowns.other_difference",
    "touchdowns.pass_difference",
    "touchdowns.total_difference",
    "touchdowns.total_return_difference",
]
