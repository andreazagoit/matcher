import NextAuth from "next-auth";
import type { User } from "@/lib/graphql/__generated__/graphql";
import type { OAuthConfig } from "next-auth/providers";

const baseUrl = process.env.NEXT_PUBLIC_APP_URL!;

/**
 * Custom OAuth Provider for Matcher
 */
const matcherProvider: OAuthConfig<User> = {
    id: "matcher",
    name: "Matcher",
    type: "oauth",
    authorization: {
        url: `${baseUrl}/oauth/authorize`,
        params: { scope: "openid profile email", response_type: "code" },
    },
    token: `${baseUrl}/oauth/api/token`,
    userinfo: `${baseUrl}/oauth/api/userinfo`,
    clientId: process.env.OAUTH_CLIENT_ID!,
    clientSecret: process.env.OAUTH_CLIENT_SECRET!,
    profile(profile: User) {
        return {
            id: profile.id,
            name: `${profile.firstName} ${profile.lastName}`,
            email: profile.email,
            firstName: profile.firstName,
            lastName: profile.lastName,
            birthDate: profile.birthDate,
            gender: profile.gender,
            createdAt: profile.createdAt,
            updatedAt: profile.updatedAt,
            image: profile.image,
        };
    },
};

export const { handlers, signIn, signOut, auth } = NextAuth({
    providers: [matcherProvider],
    session: {
        strategy: "jwt",
        maxAge: 7 * 24 * 60 * 60, // 7 days
    },
    callbacks: {
        async jwt({ token, user, account }) {
            // First time sign in
            if (user) {
                const u = user as unknown as User;
                token.id = u.id;
                token.firstName = u.firstName;
                token.lastName = u.lastName;
                token.email = u.email;
                token.birthDate = u.birthDate;
                token.gender = u.gender;
                token.createdAt = u.createdAt;
                token.updatedAt = u.updatedAt;
                token.image = u.image;
            }
            if (account) {
                token.accessToken = account.access_token;
            }
            return token;
        },
        async session({ session, token }) {
            if (token && session.user) {
                const user = session.user as unknown as User;
                session.user.id = token.id as string;
                user.firstName = token.firstName as string;
                user.lastName = token.lastName as string;
                user.birthDate = token.birthDate as string;
                user.gender = token.gender as User['gender'];
                user.createdAt = token.createdAt as string;
                user.updatedAt = token.updatedAt as string;
                user.image = token.image as User['image'];
            }
            return session;
        },
    },
    trustHost: true,
});
