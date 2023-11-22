# constants.py

# List of column names to keep
COLUMNS_TO_KEEP = [
    "odds.spread_close",
    "passing.totals.air_yards_difference",
    "passing.totals.attempts_difference",
    "passing.totals.avg_yards_ratio",
    "passing.totals.blitzes_difference",
    "passing.totals.cmp_pct_ratio",
    "passing.totals.completions_difference",
    "passing.totals.defended_passes_difference",
    "passing.totals.dropped_passes_difference",
    "passing.totals.hurries_difference",
    "passing.totals.interceptions_difference",
    "passing.totals.knockdowns_difference",
    "passing.totals.longest_difference",
    "passing.totals.net_yards_difference",
    "passing.totals.pocket_time_difference",
    "passing.totals.rating_ratio",
    "passing.totals.redzone_attempts_difference",
    "passing.totals.sack_yards_difference",
    "passing.totals.sacks_difference",
    "passing.totals.spikes_difference",
    "passing.totals.throw_aways_difference",
    "passing.totals.touchdowns_difference",
    "passing.totals.yards_difference",
    "rushing.totals.attempts_difference",
    "rushing.totals.avg_yards_ratio",
    "rushing.totals.broken_tackles_difference",
    "rushing.totals.kneel_downs_difference",
    "rushing.totals.longest_difference",
    "rushing.totals.redzone_attempts_difference",
    "rushing.totals.scrambles_difference",
    "rushing.totals.tlost_difference",
    "rushing.totals.tlost_yards_difference",
    "rushing.totals.touchdowns_difference",
    "rushing.totals.yards_difference",
    "rushing.totals.yards_after_contact_difference",
    "receiving.totals.air_yards_difference",
    "receiving.totals.avg_yards_ratio",
    "receiving.totals.broken_tackles_difference",
    "receiving.totals.catchable_passes_difference",
    "receiving.totals.dropped_passes_difference",
    "receiving.totals.longest_difference",
    "receiving.totals.receptions_difference",
    "receiving.totals.redzone_targets_difference",
    "receiving.totals.targets_difference",
    "receiving.totals.touchdowns_difference",
    "receiving.totals.yards_difference",
    "receiving.totals.yards_after_catch_ratio",
    "receiving.totals.yards_after_contact_difference",
    "summary.avg_gain_ratio",
    "summary.fumbles_difference",
    "summary.lost_fumbles_difference",
    "summary.penalties_difference",
    "summary.penalty_yards_difference",
    "summary.play_count_difference",
    "summary.possession_time_difference",
    "summary.return_yards_difference",
    "summary.rush_plays_difference",
    "summary.safeties_difference",
    "summary.total_yards_difference",
    "summary.turnovers_difference",
    "efficiency.fourthdown.attempts_difference",
    "efficiency.fourthdown.pct_ratio",
    "efficiency.fourthdown.successes_difference",
    "efficiency.goaltogo.attempts_difference",
    "efficiency.goaltogo.successes_difference",
    "efficiency.redzone.attempts_difference",
    "efficiency.redzone.successes_difference",
    "efficiency.thirddown.attempts_difference",
    "efficiency.thirddown.pct_ratio",
    "efficiency.thirddown.successes_difference",
    "defense.totals.assists_difference",
    "defense.totals.blitzes_difference",
    "defense.totals.combined_difference",
    "defense.totals.def_comps_difference",
    "defense.totals.def_targets_difference",
    "defense.totals.forced_fumbles_difference",
    "defense.totals.fumble_recoveries_difference",
    "defense.totals.hurries_difference",
    "defense.totals.interceptions_difference",
    "defense.totals.knockdowns_difference",
    "defense.totals.misc_assists_difference",
    "defense.totals.misc_forced_fumbles_difference",
    "defense.totals.misc_fumble_recoveries_difference",
    "defense.totals.misc_tackles_difference",
    "defense.totals.missed_tackles_difference",
    "defense.totals.passes_defended_difference",
    "defense.totals.qb_hits_difference",
    "defense.totals.sack_yards_difference",
    "defense.totals.sacks_difference",
    "defense.totals.safeties_difference",
    "defense.totals.sp_assists_difference",
    "defense.totals.sp_blocks_difference",
    "defense.totals.sp_forced_fumbles_difference",
    "defense.totals.sp_fumble_recoveries_difference",
    "defense.totals.sp_tackles_difference",
    "defense.totals.tackles_difference",
    "defense.totals.tloss_difference",
    "defense.totals.tloss_yards_difference",
    "advanced.air_to_total_yards_rate_difference",
    "advanced.defensive_line_rating_difference",
    "advanced.drop_rate_difference",
    "advanced.offensive_line_rating_difference",
    "advanced.pressure_rate_on_qb_difference",
    "advanced.qb_hit_rate_difference",
    "advanced.rush_pass_rate_difference",
    "advanced.tackle_for_loss_rate_difference",
    "extra_points.conversions.totals.defense_attempts_difference",
    "extra_points.conversions.totals.defense_successes_difference",
    "extra_points.conversions.totals.pass_attempts_difference",
    "extra_points.conversions.totals.pass_successes_difference",
    "extra_points.conversions.totals.rush_attempts_difference",
    "extra_points.conversions.totals.rush_successes_difference",
    "extra_points.kicks.totals.attempts_difference",
    "extra_points.kicks.totals.blocked_difference",
    "extra_points.kicks.totals.made_difference",
    "extra_points.kicks.totals.pct_difference",
    "field_goals.totals.attempts_difference",
    "field_goals.totals.avg_yards_difference",
    "field_goals.totals.blocked_difference",
    "field_goals.totals.made_difference",
    "field_goals.totals.net_attempts_difference",
    "field_goals.totals.pct_ratio",
    "field_goals.totals.yards_difference",
    "first_downs.pass_difference",
    "first_downs.penalty_difference",
    "first_downs.rush_difference",
    "first_downs.total_difference",
    "fumbles.totals.ez_rec_tds_difference",
    "fumbles.totals.forced_fumbles_difference",
    "fumbles.totals.fumbles_difference",
    "fumbles.totals.lost_fumbles_difference",
    "fumbles.totals.opp_rec_difference",
    "fumbles.totals.opp_rec_tds_difference",
    "fumbles.totals.opp_rec_yards_difference",
    "fumbles.totals.out_of_bounds_difference",
    "fumbles.totals.own_rec_difference",
    "fumbles.totals.own_rec_tds_difference",
    "fumbles.totals.own_rec_yards_difference",
    "int_returns.totals.avg_yards_difference",
    "int_returns.totals.number_difference",
    "int_returns.totals.touchdowns_difference",
    "int_returns.totals.yards_difference",
    "interceptions.number_difference",
    "interceptions.return_yards_difference",
    "interceptions.returned_difference",
    "kick_returns.totals.avg_yards_difference",
    "kick_returns.totals.faircatches_difference",
    "kick_returns.totals.number_difference",
    "kick_returns.totals.touchdowns_difference",
    "kick_returns.totals.yards_difference",
    "kickoffs.totals.endzone_difference",
    "kickoffs.totals.inside_20_difference",
    "kickoffs.totals.number_difference",
    "kickoffs.totals.onside_attempts_difference",
    "kickoffs.totals.onside_successes_difference",
    "kickoffs.totals.out_of_bounds_difference",
    "kickoffs.totals.return_yards_difference",
    "kickoffs.totals.squib_kicks_difference",
    "kickoffs.totals.total_endzone_difference",
    "kickoffs.totals.touchbacks_difference",
    "kickoffs.totals.yards_difference",
    "misc_returns.totals.blk_fg_touchdowns_difference",
    "misc_returns.totals.blk_punt_touchdowns_difference",
    "misc_returns.totals.ez_rec_touchdowns_difference",
    "misc_returns.totals.fg_return_touchdowns_difference",
    "misc_returns.totals.number_difference",
    "misc_returns.totals.touchdowns_difference",
    "misc_returns.totals.yards_difference",
    "penalties.totals.penalties_difference",
    "penalties.totals.yards_difference",
    "punt_returns.totals.avg_yards_difference",
    "punt_returns.totals.faircatches_difference",
    "punt_returns.totals.longest_difference",
    "punt_returns.totals.number_difference",
    "punt_returns.totals.touchdowns_difference",
    "punt_returns.totals.yards_difference",
    "punts.totals.attempts_difference",
    "punts.totals.avg_hang_time_ratio",
    "punts.totals.avg_net_yards_ratio",
    "punts.totals.avg_yards_ratio",
    "punts.totals.blocked_difference",
    "punts.totals.hang_time_difference",
    "punts.totals.inside_20_difference",
    "punts.totals.longest_difference",
    "punts.totals.net_yards_difference",
    "punts.totals.return_yards_difference",
    "punts.totals.touchbacks_difference",
    "punts.totals.yards_difference",
    "touchdowns.fumble_return_difference",
    "touchdowns.int_return_difference",
    "touchdowns.kick_return_difference",
    "touchdowns.other_difference",
    "touchdowns.pass_difference",
    "touchdowns.punt_return_difference",
    "touchdowns.rush_difference",
    "touchdowns.total_difference",
    "touchdowns.total_return_difference",
]
