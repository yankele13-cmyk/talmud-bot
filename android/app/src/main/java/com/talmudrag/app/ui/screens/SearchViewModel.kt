package com.talmudrag.app.ui.screens

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.talmudrag.app.data.api.TalmudApi
import com.talmudrag.app.data.model.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class SearchUiState(
    val status: StatusResponse? = null,
    val tractates: List<String> = emptyList(),
    val sedarim: List<String> = emptyList(),
    val selectedFilter: String = "",  // tractate or seder
    val filterType: String = "",      // "tractate" or "seder"
    val query: String = "",
    val isLoading: Boolean = false,
    val results: SearchResponse? = null,
    val streamingAnswer: String = "",
    val isStreaming: Boolean = false,
    val error: String? = null,
)

class SearchViewModel(
    private val api: TalmudApi = TalmudApi()
) : ViewModel() {

    private val _state = MutableStateFlow(SearchUiState())
    val state = _state.asStateFlow()

    init {
        loadStatus()
        loadTractates()
    }

    private fun loadStatus() {
        viewModelScope.launch {
            try {
                val status = api.getStatus()
                _state.value = _state.value.copy(status = status)
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    status = StatusResponse(status = "offline", message = e.message ?: "")
                )
            }
        }
    }

    private fun loadTractates() {
        viewModelScope.launch {
            try {
                val data = api.getTractates()
                _state.value = _state.value.copy(
                    tractates = data.tractates,
                    sedarim = data.sedarim,
                )
            } catch (_: Exception) {}
        }
    }

    fun setQuery(query: String) {
        _state.value = _state.value.copy(query = query)
    }

    fun setFilter(name: String, type: String) {
        _state.value = _state.value.copy(selectedFilter = name, filterType = type)
    }

    fun clearFilter() {
        _state.value = _state.value.copy(selectedFilter = "", filterType = "")
    }

    fun search() {
        val query = _state.value.query.trim()
        if (query.isEmpty()) return

        _state.value = _state.value.copy(isLoading = true, error = null, results = null, streamingAnswer = "")

        viewModelScope.launch {
            try {
                val tractate = if (_state.value.filterType == "tractate") _state.value.selectedFilter else null
                val seder = if (_state.value.filterType == "seder") _state.value.selectedFilter else null

                val response = api.search(query, tractate = tractate, seder = seder)
                _state.value = _state.value.copy(
                    isLoading = false,
                    results = response,
                    streamingAnswer = response.answer?.llm_answer ?: "",
                )
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    error = e.message ?: "Search failed",
                )
            }
        }
    }
}
