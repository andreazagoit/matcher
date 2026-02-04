import type { Metadata } from "next";
import { Geist, Geist_Mono, Inter } from "next/font/google";
import "./globals.css";
import { ApolloWrapper } from "@/lib/graphql/apollo-provider";
import { AuthProvider } from "@/components/auth-provider";
import { Header } from "@/components/header";

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Matcher",
  description: "OAuth provider & matching platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} dark`}>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <AuthProvider>
          <ApolloWrapper>
            <div className="min-h-screen bg-background">
              <Header />
              <main>{children}</main>
            </div>
          </ApolloWrapper>
        </AuthProvider>
      </body>
    </html>
  );
}
