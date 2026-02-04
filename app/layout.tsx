import type { Metadata } from "next";
import { Geist, Geist_Mono, Inter } from "next/font/google";
import "./globals.css";
import { ApolloWrapper } from "@/lib/graphql/apollo-provider";
import { AuthProvider } from "@/components/auth-provider";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList } from "@/components/ui/breadcrumb"
import { UserNav } from "@/components/user-nav"

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
            <SidebarProvider>
              <AppSidebar />
              <SidebarInset>
                <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
                  <Breadcrumb>
                    <BreadcrumbList>
                      <BreadcrumbItem>
                        <BreadcrumbLink href="/">Home</BreadcrumbLink>
                      </BreadcrumbItem>
                      {/* We can add dynamic breadcrumbs later or fetch pathname */}
                    </BreadcrumbList>
                  </Breadcrumb>
                  <div className="ml-auto px-3">
                    <UserNav />
                  </div>
                </header>
                <main className="flex-1 p-4 lg:p-6">{children}</main>
              </SidebarInset>
            </SidebarProvider>
          </ApolloWrapper>
        </AuthProvider>
      </body>
    </html>
  );
}
