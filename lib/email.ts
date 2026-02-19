import nodemailer from "nodemailer";

const transporter = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  port: Number(process.env.SMTP_PORT) || 465,
  secure: process.env.SMTP_SECURE === "true",
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS,
  },
});

const FROM_NAME = process.env.SMTP_NAME || "Matcher";
const FROM_EMAIL = process.env.SMTP_USER || "noreply@example.com";

/**
 * Send an OTP code via email.
 */
export async function sendOTPEmail(
  to: string,
  otp: string,
  type: "sign-in" | "email-verification" | "forget-password",
) {
  const subjectMap: Record<string, string> = {
    "sign-in": "Il tuo codice di accesso",
    "email-verification": "Verifica la tua email",
    "forget-password": "Reset della password",
  };

  const labelMap: Record<string, string> = {
    "sign-in": "accedere al tuo account",
    "email-verification": "verificare la tua email",
    "forget-password": "reimpostare la tua password",
  };

  const subject = subjectMap[type] || "Il tuo codice";
  const label = labelMap[type] || "completare l'operazione";

  await transporter.sendMail({
    from: `"${FROM_NAME}" <${FROM_EMAIL}>`,
    to,
    subject: `${subject} — ${FROM_NAME}`,
    text: `Il tuo codice per ${label} è: ${otp}\n\nIl codice scade tra 5 minuti.\n\nSe non hai richiesto questo codice, ignora questa email.`,
    html: `
      <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 480px; margin: 0 auto; padding: 40px 20px;">
        <div style="text-align: center; margin-bottom: 32px;">
          <div style="display: inline-block; background: #18181b; color: #fff; font-weight: 700; font-size: 18px; width: 48px; height: 48px; line-height: 48px; border-radius: 12px;">
            M
          </div>
          <h2 style="margin: 16px 0 4px; font-size: 20px; color: #18181b;">${FROM_NAME}</h2>
          <p style="margin: 0; color: #71717a; font-size: 14px;">${subject}</p>
        </div>
        <div style="background: #f4f4f5; border-radius: 16px; padding: 32px; text-align: center;">
          <p style="margin: 0 0 16px; color: #3f3f46; font-size: 15px;">
            Usa questo codice per ${label}:
          </p>
          <div style="font-size: 36px; font-weight: 700; letter-spacing: 0.3em; color: #18181b; font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace; padding: 12px 0;">
            ${otp}
          </div>
          <p style="margin: 16px 0 0; color: #a1a1aa; font-size: 13px;">
            Il codice scade tra 5 minuti
          </p>
        </div>
        <p style="margin: 24px 0 0; color: #a1a1aa; font-size: 12px; text-align: center;">
          Se non hai richiesto questo codice, puoi ignorare questa email.
        </p>
      </div>
    `,
  });
}
