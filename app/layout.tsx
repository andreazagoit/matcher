"use client";

import { Geist, Geist_Mono, Inter } from "next/font/google";
import "./globals.css";
import { ApolloWrapper } from "@/lib/graphql/apollo-provider";
import { AuthProvider } from "@/components/auth-provider";
import { usePathname } from "next/navigation";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const AUTH_ROUTES = ["/sign-in", "/sign-up"];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const isAuthRoute = AUTH_ROUTES.some(
    (r) => pathname === r || pathname.startsWith(r + "/"),
  );

  return (
    <html lang="en" className={`${inter.variable} dark`} suppressHydrationWarning>
      <head>
        <title>Matcher</title>
        <meta name="description" content="Friendship matching platform" />
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <AuthProvider>
          <ApolloWrapper>
            {isAuthRoute ? (
              <div className="min-h-screen flex flex-col">{children}</div>
            ) : (
              <SidebarProvider>
                <AppSidebar />
                <SidebarInset>
                  <div className="md:hidden sticky top-0 z-20 border-b bg-background/95 backdrop-blur px-2 py-2">
                    <SidebarTrigger />
                  </div>
                  {children}
                </SidebarInset>
              </SidebarProvider>
            )}
          </ApolloWrapper>
        </AuthProvider>
      </body>
    </html>
  );
}
