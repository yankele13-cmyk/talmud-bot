package com.talmudrag.app.data.api

import com.talmudrag.app.BuildConfig
import com.talmudrag.app.data.model.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okhttp3.*
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import java.io.IOException
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class TalmudApi(
    private val baseUrl: String = BuildConfig.API_BASE_URL
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(120, java.util.concurrent.TimeUnit.SECONDS)
        .build()

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
    }

    suspend fun getStatus(): StatusResponse = withContext(Dispatchers.IO) {
        val request = Request.Builder().url("$baseUrl/api/status").build()
        val response = client.newCall(request).await()
        json.decodeFromString(response.body!!.string())
    }

    suspend fun getTractates(): TractatesResponse = withContext(Dispatchers.IO) {
        val request = Request.Builder().url("$baseUrl/api/tractates").build()
        val response = client.newCall(request).await()
        json.decodeFromString(response.body!!.string())
    }

    suspend fun search(
        query: String,
        topK: Int = 5,
        tractate: String? = null,
        seder: String? = null,
    ): SearchResponse = withContext(Dispatchers.IO) {
        val url = "$baseUrl/api/search".toHttpUrl().newBuilder().apply {
            addQueryParameter("q", query)
            addQueryParameter("top_k", topK.toString())
            tractate?.let { addQueryParameter("tractate", it) }
            seder?.let { addQueryParameter("seder", it) }
        }.build()

        val request = Request.Builder().url(url).build()
        val response = client.newCall(request).await()
        json.decodeFromString(response.body!!.string())
    }

    fun searchStream(
        query: String,
        topK: Int = 5,
        tractate: String? = null,
        onSources: (SearchResponse) -> Unit,
        onToken: (String) -> Unit,
        onDone: () -> Unit,
        onError: (Exception) -> Unit,
    ): EventSource {
        val url = "$baseUrl/api/search/stream".toHttpUrl().newBuilder().apply {
            addQueryParameter("q", query)
            addQueryParameter("top_k", topK.toString())
            tractate?.let { addQueryParameter("tractate", it) }
        }.build()

        val request = Request.Builder().url(url).build()

        val listener = object : EventSourceListener() {
            override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                if (data == "[DONE]") {
                    onDone()
                    return
                }
                try {
                    val event = json.decodeFromString<Map<String, kotlinx.serialization.json.JsonElement>>(data)
                    val eventType = event["type"]?.toString()?.removeSurrounding("\"") ?: ""
                    when (eventType) {
                        "sources" -> {
                            val sourceData = event["data"]?.toString() ?: "{}"
                            onSources(json.decodeFromString(sourceData))
                        }
                        "token" -> {
                            val text = event["text"]?.toString()?.removeSurrounding("\"") ?: ""
                            onToken(text)
                        }
                    }
                } catch (e: Exception) {
                    // Ignore parse errors for partial data
                }
            }

            override fun onFailure(eventSource: EventSource, t: Throwable?, response: Response?) {
                onError(Exception(t?.message ?: "Stream failed"))
            }
        }

        return EventSources.createFactory(client).newEventSource(request, listener)
    }

    private suspend fun Call.await(): Response = suspendCancellableCoroutine { cont ->
        enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) cont.resume(response)
                else cont.resumeWithException(IOException("HTTP ${response.code}"))
            }
            override fun onFailure(call: Call, e: IOException) {
                cont.resumeWithException(e)
            }
        })
        cont.invokeOnCancellation { cancel() }
    }
}
