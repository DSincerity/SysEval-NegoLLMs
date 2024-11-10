"""
A registry to define the supported datasets, models, and tasks.
"""

# the new tasks organized - first by datasets and then based on whether it was defined by Emily or Tara.
SUPPORTED_CONFIGS = set([

    ("dnd", "open_ai", "sta_total_item_count_dnd"),
    ("dnd", "open_ai", "sta_max_points_dnd"),
    ("dnd", "open_ai", "mid_dial_act_dnd"),
    ("dnd", "open_ai", "mid_gen_resp_dnd"),
    ("dnd", "open_ai", "end_deal_specifics_dnd"),

    ("dnd", "open_ai", "sta_ask_point_values_dnd"),
    ("dnd", "open_ai", "mid_full_proposal_dnd"),
    ("dnd", "open_ai", "end_deal_total_dnd"),

    ("dnd", "hf_model", "sta_total_item_count_dnd"),
    ("dnd", "hf_model", "sta_max_points_dnd"),
    ("dnd", "hf_model", "mid_dial_act_dnd"),
    ("dnd", "hf_model", "mid_gen_resp_dnd"),
    ("dnd", "hf_model", "end_deal_specifics_dnd"),

    ("dnd", "hf_model", "sta_ask_point_values_dnd"),
    ("dnd", "hf_model", "mid_full_proposal_dnd"),
    ("dnd", "hf_model", "end_deal_total_dnd"),

    ("casino", "open_ai", "sta_total_item_count_ca"),
    ("casino", "open_ai", "mid_strategy_ca"),
    ("casino", "open_ai", "mid_gen_resp_ca"),
    ("casino", "open_ai", "end_deal_specifics_ca"),
    ("casino", "open_ai", "end_deal_total_ca"),

    ("casino", "open_ai", "sta_max_points_ca"),
    ("casino", "open_ai", "sta_ask_point_values_ca"),
    ("casino", "open_ai", "sta_ask_low_priority_ca"),
    ("casino", "open_ai", "sta_ask_high_priority_ca"),
    ("casino", "open_ai", "mid_ask_low_priority_ca"),
    ("casino", "open_ai", "mid_ask_high_priority_ca"),
    ("casino", "open_ai", "mid_partner_ask_low_priority_ca"),
    ("casino", "open_ai", "mid_partner_ask_high_priority_ca"),
    ("casino", "open_ai", "end_deal_likeness_ca"),
    ("casino", "open_ai", "end_deal_satisfaction_ca"),
    ("casino", "open_ai", "end_partner_deal_likeness_ca"),
    ("casino", "open_ai", "end_partner_deal_satisfaction_ca"),

    ("casino", "hf_model", "sta_total_item_count_ca"),
    ("casino", "hf_model", "mid_strategy_ca"),
    ("casino", "hf_model", "mid_gen_resp_ca"),
    ("casino", "hf_model", "end_deal_specifics_ca"),
    ("casino", "hf_model", "end_deal_total_ca"),

    ("casino", "hf_model", "sta_max_points_ca"),
    ("casino", "hf_model", "sta_ask_point_values_ca"),
    ("casino", "hf_model", "sta_ask_low_priority_ca"),
    ("casino", "hf_model", "sta_ask_high_priority_ca"),
    ("casino", "hf_model", "mid_ask_low_priority_ca"),
    ("casino", "hf_model", "mid_ask_high_priority_ca"),
    ("casino", "hf_model", "mid_partner_ask_low_priority_ca"),
    ("casino", "hf_model", "mid_partner_ask_high_priority_ca"),
    ("casino", "hf_model", "end_deal_likeness_ca"),
    ("casino", "hf_model", "end_deal_satisfaction_ca"),
    ("casino", "hf_model", "end_partner_deal_likeness_ca"),
    ("casino", "hf_model", "end_partner_deal_satisfaction_ca"),

    ("job_interview", "open_ai", "end_deal_specifics_ji"),

    # ASSUME THE BOT IS A WORKER
    ("job_interview", "open_ai", "sta_ask_high_priority_ji_w"),
    ("job_interview", "open_ai", "sta_ask_low_priority_ji_w"),
    ("job_interview", "open_ai", "mid_ask_high_priority_ji_w"),
    ("job_interview", "open_ai", "mid_ask_low_priority_ji_w"),
    ("job_interview", "open_ai", "mid_partner_ask_high_priority_ji_w"),
    ("job_interview", "open_ai", "mid_partner_ask_low_priority_ji_w"),
    ("job_interview", "open_ai", "mid_dial_act_ji"),

    ("cra", "open_ai", "mid_dial_act_cra"),
    ("cra", "open_ai", "mid_full_proposal_cra"),

    ("job_interview", "hf_model", "end_deal_specifics_ji"),

    # ASSUME THE BOT IS A WORKER
    ("job_interview", "hf_model", "sta_ask_high_priority_ji_w"),
    ("job_interview", "hf_model", "sta_ask_low_priority_ji_w"),
    ("job_interview", "hf_model", "mid_ask_high_priority_ji_w"),
    ("job_interview", "hf_model", "mid_ask_low_priority_ji_w"),
    ("job_interview", "hf_model", "mid_partner_ask_high_priority_ji_w"),
    ("job_interview", "hf_model", "mid_partner_ask_low_priority_ji_w"),
    ("job_interview", "hf_model", "mid_dial_act_ji"),

    ("cra", "hf_model", "mid_dial_act_cra"),
    ("cra", "hf_model", "mid_full_proposal_cra"),

])


CLS_NAME2PATHS = {
    "nego_datasets": {
        "dnd": "nego_datasets.dealornodeal.DNDHandler",
        "casino": "nego_datasets.casino.CasinoHandler",
        "job_interview": "nego_datasets.jobinterview.JIHandler",
        "cra": "nego_datasets.cra.CRAHandler"
    },

    "models": {
        "open_ai": "models.open_ai.OpenAIHandler",
        "hf_model": "models.hf_model.HFModelHandler",
        "llama_7b": "models.llama.Llama7BHandler",
        "falcon_7b": "models.falcon.Falcon7BHandler",
    },

    "tasks": {
        "sta_total_item_count_dnd": "tasks.sta_total_item_count_dnd.TICNDHandlerDND",
        "sta_max_points_dnd": "tasks.sta_max_points_dnd.A1MPNDHandlerDND",
        "mid_dial_act_dnd": "tasks.mid_dial_act_dnd.DASUHandler",
        "mid_gen_resp_dnd": "tasks.mid_gen_resp_dnd.GSDNDHandler",
        "end_was_deal_achieved_dnd": "tasks.end_was_deal_achieved_dnd.DHandlerDND",
        "end_deal_specifics_dnd": "tasks.end_deal_specifics_dnd.A1BCHandler",

        "sta_ask_point_values_dnd": "tasks.sta_ask_point_values_dnd.BYDNDNDPointValuesHandler",
        "mid_full_proposal_dnd": "tasks.mid_full_proposal_dnd.DNDRegAllSlotsHandler",
        "end_deal_total_dnd": "tasks.end_deal_total_dnd.YDNDDealPointsHandler",

        "sta_total_item_count_ca": "tasks.sta_total_item_count_ca.TICHandlerCa",
        "mid_strategy_ca": "tasks.mid_strategy_ca.NSUHandler",
        "mid_gen_resp_ca": "tasks.mid_gen_resp_ca.GSCaHandler",
        "end_was_deal_achieved_ca": "tasks.end_was_deal_achieved_ca.DHandlerCa",
        "end_deal_specifics_ca": "tasks.end_deal_specifics_ca.A1FCHandler",
        "end_deal_total_ca": "tasks.end_deal_total_ca.A1PHandlerCa",

        "sta_max_points_ca": "tasks.sta_max_points_ca.A1CaNDMaxPointsHandler",
        "sta_ask_point_values_ca": "tasks.sta_ask_point_values_ca.Food1CaNDPointValuesHandler",
        "sta_ask_low_priority_ca": "tasks.sta_ask_low_priority_ca.Low1CaWCPrioritiesHandler",
        "sta_ask_high_priority_ca": "tasks.sta_ask_high_priority_ca.High1CaWCPrioritiesHandler",
        "mid_ask_low_priority_ca": "tasks.mid_ask_low_priority_ca.MidLow1CaWCPrioritiesHandler",
        "mid_ask_high_priority_ca": "tasks.mid_ask_high_priority_ca.MidHigh1CaWCPrioritiesHandler",
        "mid_partner_ask_low_priority_ca": "tasks.mid_partner_ask_low_priority_ca.MidPartnerLow1CaWCPrioritiesHandler",
        "mid_partner_ask_high_priority_ca": "tasks.mid_partner_ask_high_priority_ca.MidPartnerHigh1CaWCPrioritiesHandler",
        "end_deal_likeness_ca": "tasks.end_deal_likeness_ca.A1LikenessCAHandler",
        "end_deal_satisfaction_ca": "tasks.end_deal_satisfaction_ca.A1SatisfactionCAHandler",
        "end_partner_deal_likeness_ca": "tasks.end_partner_deal_likeness_ca.A1PartnerLikenessCAHandler",
        "end_partner_deal_satisfaction_ca": "tasks.end_partner_deal_satisfaction_ca.A1PartnerSatisfactionCAHandler",

        "end_was_deal_achieved_ji": "tasks.end_was_deal_achieved_ji.DHandlerJI",
        "end_deal_specifics_ji": "tasks.end_deal_specifics_ji.FComHandler",

        "sta_ask_high_priority_ji_w": "tasks.sta_ask_high_priority_ji_w.WHighJIPrioritiesHandler",
        "sta_ask_low_priority_ji_w": "tasks.sta_ask_low_priority_ji_w.WLowJIPrioritiesHandler",
        "mid_ask_high_priority_ji_w": "tasks.mid_ask_high_priority_ji_w.WMidHighJIPrioritiesHandler",
        "mid_ask_low_priority_ji_w": "tasks.mid_ask_low_priority_ji_w.WMidLowJIPrioritiesHandler",
        "mid_partner_ask_high_priority_ji_w": "tasks.mid_partner_ask_high_priority_ji_w.WMidPartnerHighJIPrioritiesHandler",
        "mid_partner_ask_low_priority_ji_w": "tasks.mid_partner_ask_low_priority_ji_w.WMidPartnerLowJIPrioritiesHandler",
        "mid_dial_act_ji": "tasks.mid_dial_act_ji.JIRegDAHandler",

        "mid_dial_act_cra": "tasks.mid_dial_act_cra.CRARegDAHandler",
        "mid_full_proposal_cra": "tasks.mid_full_proposal_cra.CRARegAllSlotsHandler",

    }
}
