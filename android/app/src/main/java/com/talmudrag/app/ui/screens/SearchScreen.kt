package com.talmudrag.app.ui.screens

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.talmudrag.app.data.model.Source
import com.talmudrag.app.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchScreen(viewModel: SearchViewModel = viewModel()) {
    val state by viewModel.state.collectAsState()

    Column(modifier = Modifier.fillMaxSize().background(Background)) {
        // Header
        Header(state)

        // Search bar
        SearchBar(
            query = state.query,
            onQueryChange = viewModel::setQuery,
            onSearch = viewModel::search,
            isLoading = state.isLoading,
        )

        // Filter chips
        FilterChips(
            tractates = state.tractates,
            sedarim = state.sedarim,
            selected = state.selectedFilter,
            onSelect = viewModel::setFilter,
            onClear = viewModel::clearFilter,
        )

        // Quick questions
        if (state.results == null && !state.isLoading) {
            QuickQuestions(
                onSelect = { q ->
                    viewModel.setQuery(q)
                    viewModel.search()
                }
            )
        }

        // Results
        when {
            state.isLoading -> LoadingView()
            state.error != null -> ErrorView(state.error!!)
            state.results != null -> ResultsList(state)
            else -> EmptyState()
        }
    }
}

@Composable
private fun Header(state: SearchUiState) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Surface)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Talmud RAG", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Accent)
        Text("Gemara + Rashi + Tosafot", fontSize = 12.sp, color = TextDim)

        Spacer(Modifier.height(6.dp))

        val (badgeColor, badgeText) = when (state.status?.status) {
            "ready" -> GemaraGreen to "${state.status.version ?: "v5"} | ${state.status.chunks} chunks"
            "loading" -> Accent to "Loading..."
            else -> TosafotRed to (state.status?.message ?: "Offline")
        }

        Surface(
            shape = RoundedCornerShape(20.dp),
            color = badgeColor.copy(alpha = 0.15f),
        ) {
            Text(badgeText, color = badgeColor, fontSize = 11.sp,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 2.dp))
        }
    }

    HorizontalDivider(thickness = 2.dp, color = Accent)
}

@Composable
private fun SearchBar(
    query: String,
    onQueryChange: (String) -> Unit,
    onSearch: () -> Unit,
    isLoading: Boolean,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(Surface)
            .padding(12.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedTextField(
            value = query,
            onValueChange = onQueryChange,
            placeholder = { Text("...חפש בתלמוד", color = TextDim) },
            modifier = Modifier.weight(1f),
            shape = RoundedCornerShape(12.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Accent,
                unfocusedBorderColor = Border,
                cursorColor = Accent,
            ),
            singleLine = true,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
            keyboardActions = KeyboardActions(onSearch = { onSearch() }),
            textStyle = LocalTextStyle.current.copy(textAlign = TextAlign.End),
        )

        Button(
            onClick = onSearch,
            enabled = !isLoading && query.isNotBlank(),
            colors = ButtonDefaults.buttonColors(containerColor = Accent),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.height(56.dp),
        ) {
            if (isLoading) {
                CircularProgressIndicator(modifier = Modifier.size(20.dp), color = Color.Black, strokeWidth = 2.dp)
            } else {
                Icon(Icons.Default.Search, "Search", tint = Color.Black)
            }
        }
    }
}

@Composable
private fun FilterChips(
    tractates: List<String>,
    sedarim: List<String>,
    selected: String,
    onSelect: (String, String) -> Unit,
    onClear: () -> Unit,
) {
    LazyRow(
        modifier = Modifier
            .fillMaxWidth()
            .background(Surface)
            .padding(horizontal = 12.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        item {
            FilterChipItem("All", selected.isEmpty(), Accent) { onClear() }
        }
        items(sedarim) { seder ->
            FilterChipItem(seder, selected == seder, Accent) { onSelect(seder, "seder") }
        }
        items(tractates) { tractate ->
            FilterChipItem(tractate, selected == tractate, TextPrimary) { onSelect(tractate, "tractate") }
        }
    }

    HorizontalDivider(thickness = 1.dp, color = Border)
}

@Composable
private fun FilterChipItem(label: String, isSelected: Boolean, activeColor: Color, onClick: () -> Unit) {
    Surface(
        onClick = onClick,
        shape = RoundedCornerShape(20.dp),
        color = if (isSelected) activeColor else Surface2,
        border = if (!isSelected) BorderStroke(1.dp, Border) else null,
    ) {
        Text(
            text = label,
            color = if (isSelected) Color.Black else TextDim,
            fontSize = 13.sp,
            fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 6.dp),
        )
    }
}

@Composable
private fun QuickQuestions(onSelect: (String) -> Unit) {
    val questions = listOf(
        "מהו הזמן של קריאת שמע?",
        "גאולה לתפילה",
        "ל״ט מלאכות שבת",
        "שור שנגח",
        "כתובה של אשה",
    )

    LazyRow(
        modifier = Modifier.padding(12.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items(questions) { q ->
            Surface(
                onClick = { onSelect(q) },
                shape = RoundedCornerShape(12.dp),
                color = Surface2,
                border = BorderStroke(1.dp, Border),
            ) {
                Text(q, color = TextPrimary, fontSize = 13.sp,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp))
            }
        }
    }
}

@Composable
private fun LoadingView() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator(color = Accent)
            Spacer(Modifier.height(16.dp))
            Text("...מחפש בש״ס", color = TextDim)
        }
    }
}

@Composable
private fun ErrorView(error: String) {
    Box(modifier = Modifier.fillMaxSize().padding(32.dp), contentAlignment = Alignment.Center) {
        Text(error, color = TosafotRed, textAlign = TextAlign.Center)
    }
}

@Composable
private fun EmptyState() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("📜", fontSize = 48.sp)
            Spacer(Modifier.height(16.dp))
            Text("Search the Talmud Bavli", color = TextDim)
            Text("Gemara, Rashi & Tosafot indexed", color = TextDim, fontSize = 13.sp)
        }
    }
}

@Composable
private fun ResultsList(state: SearchUiState) {
    val answer = state.results?.answer ?: return

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        contentPadding = PaddingValues(vertical = 12.dp),
    ) {
        // Time info
        item {
            Text(
                "${answer.total} results from ${state.results.candidates_evaluated} candidates in ${state.results.time_ms}ms",
                color = TextDim, fontSize = 13.sp,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center,
            )
        }

        // LLM Answer card
        answer.llm_answer?.let { ravAnswer ->
            item { RavAnswerCard(ravAnswer) }
        }

        // Source cards
        items(answer.sources) { source ->
            SourceCard(source)
        }
    }
}

@Composable
private fun RavAnswerCard(answer: String) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = Surface,
        border = BorderStroke(2.dp, Accent),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("RAV (Claude AI)", color = Accent, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            Text(answer, color = TextPrimary, fontSize = 15.sp, lineHeight = 24.sp)
        }
    }
}

@Composable
private fun SourceCard(source: Source) {
    val context = LocalContext.current
    var expanded by remember { mutableStateOf(false) }

    Surface(
        shape = RoundedCornerShape(12.dp),
        color = Surface,
        border = BorderStroke(1.dp, if (source.rank == 1) Accent else Border),
    ) {
        Column {
            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Surface2)
                    .padding(12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(source.ref, color = Accent, fontWeight = FontWeight.Bold, fontSize = 15.sp)
                Surface(shape = RoundedCornerShape(10.dp), color = Background) {
                    Text("${source.rerank_score.format(2)}", color = TextDim, fontSize = 12.sp,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp))
                }
            }

            // Badges
            Row(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                if (source.tractate.isNotEmpty()) Badge(source.tractate, Accent)
                if (source.seder.isNotEmpty()) Badge(source.seder, TextDim)
                Badge("Gemara", GemaraGreen)
                if (source.has_rashi) Badge("Rashi", RashiBlue)
                if (source.has_tosafot) Badge("Tosafot", TosafotRed)
            }

            // Key passages
            if (source.key_passages.isNotEmpty()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Accent.copy(alpha = 0.05f))
                        .padding(12.dp)
                ) {
                    Text("KEY PASSAGES", color = Accent, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    source.key_passages.take(2).forEach { passage ->
                        Text(
                            passage.take(200),
                            color = TextPrimary, fontSize = 13.sp,
                            modifier = Modifier.padding(top = 4.dp),
                            maxLines = 3, overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }

            // Full text (expandable)
            Text(
                text = source.full_text,
                color = TextPrimary, fontSize = 15.sp, lineHeight = 24.sp,
                modifier = Modifier.padding(12.dp),
                maxLines = if (expanded) Int.MAX_VALUE else 6,
                overflow = TextOverflow.Ellipsis,
            )

            // Footer
            HorizontalDivider(color = Border)
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                TextButton(onClick = {
                    if (source.url.isNotEmpty()) {
                        context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(source.url)))
                    }
                }) {
                    Text("Sefaria →", color = RashiBlue, fontSize = 13.sp)
                }

                TextButton(onClick = { expanded = !expanded }) {
                    Text(if (expanded) "Collapse" else "Expand", color = TextDim, fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
private fun Badge(text: String, color: Color) {
    Surface(
        shape = RoundedCornerShape(10.dp),
        color = color.copy(alpha = 0.15f),
    ) {
        Text(text, color = color, fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp))
    }
}

private fun Double.format(digits: Int) = "%.${digits}f".format(this)
