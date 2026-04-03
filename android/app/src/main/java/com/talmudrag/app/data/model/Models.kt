package com.talmudrag.app.data.model

import kotlinx.serialization.Serializable

@Serializable
data class SearchResponse(
    val query: String = "",
    val expanded: String = "",
    val hyde_used: Boolean = false,
    val answer: AnswerData? = null,
    val time_ms: Int = 0,
    val candidates_evaluated: Int = 0,
)

@Serializable
data class AnswerData(
    val summary: String = "",
    val llm_answer: String? = null,
    val sources: List<Source> = emptyList(),
    val total: Int = 0,
)

@Serializable
data class Source(
    val rank: Int = 0,
    val ref: String = "",
    val tractate: String = "",
    val seder: String = "",
    val url: String = "",
    val rerank_score: Double = 0.0,
    val bi_score: Double = 0.0,
    val has_gemara: Boolean = false,
    val has_rashi: Boolean = false,
    val has_tosafot: Boolean = false,
    val full_text: String = "",
    val key_passages: List<String> = emptyList(),
)

@Serializable
data class StatusResponse(
    val status: String = "",
    val version: String = "",
    val chunks: Int = 0,
    val features: List<String> = emptyList(),
    val message: String = "",
)

@Serializable
data class TractatesResponse(
    val tractates: List<String> = emptyList(),
    val sedarim: List<String> = emptyList(),
)
