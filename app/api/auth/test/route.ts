/**
 * Auth Test API
 * GET /api/auth/test
 * 
 * Tests Auth.js server-side auth() function
 */

import { auth } from "@/lib/auth";

export async function GET() {
    const session = await auth();

    return Response.json({
        serverSideAuth: {
            hasSession: !!session,
            user: session?.user || null,
        },
        timestamp: new Date().toISOString(),
    });
}
