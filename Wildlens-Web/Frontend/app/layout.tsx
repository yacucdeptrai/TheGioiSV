import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Header from "../components/Header";
import Footer from "../components/Footer";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "WildLens • Wildlife Detection",
  description: "Detect wildlife in your photos using the WildLens AI model.",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
  keywords: ["wildlife", "detection", "AI", "YOLO", "computer vision"],
  applicationName: "WildLens",
  authors: [{ name: "WildLens" }],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0ea5e9",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>        
        <Header />
        <main>
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
