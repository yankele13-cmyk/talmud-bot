package com.talmudrag.app.ui.theme

import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val Background = Color(0xFF0F0F1A)
val Surface = Color(0xFF1A1A2E)
val Surface2 = Color(0xFF252540)
val Accent = Color(0xFFE2B714)
val Accent2 = Color(0xFFC49B0E)
val TextPrimary = Color(0xFFE8E8E8)
val TextDim = Color(0xFF8888AA)
val RashiBlue = Color(0xFF4A9EFF)
val TosafotRed = Color(0xFFFF6B6B)
val GemaraGreen = Color(0xFF50FA7B)
val Border = Color(0xFF333355)

private val DarkColors = darkColorScheme(
    primary = Accent,
    onPrimary = Color.Black,
    secondary = Accent2,
    background = Background,
    surface = Surface,
    surfaceVariant = Surface2,
    onBackground = TextPrimary,
    onSurface = TextPrimary,
    onSurfaceVariant = TextDim,
    outline = Border,
)

@Composable
fun TalmudTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColors,
        typography = Typography(),
        content = content,
    )
}
