import type { Metadata } from "next";
import { Geist, Geist_Mono, Inter } from "next/font/google";
import "./globals.css";
import { ApolloWrapper } from "@/lib/graphql/apollo-provider";
import { AuthProvider } from "@/components/auth-provider";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"

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
    <html lang="en" className={`${inter.variable} dark`} suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <AuthProvider>
          <ApolloWrapper>
            <SidebarProvider>
              <AppSidebar />
              <SidebarInset>
                <div className="md:hidden sticky top-0 z-20 border-b bg-background/95 backdrop-blur px-2 py-2">
                  <SidebarTrigger />
                </div>
                {children}
              </SidebarInset>
            </SidebarProvider>
          </ApolloWrapper>
        </AuthProvider>
      </body>
    </html>
  );
}
